from typing import List

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn as nn
from torch.quasirandom import SobolEngine
from tqdm import tqdm


def my_kmeans(
    x: torch.Tensor,
    k: int,
    iters: int,
    means: torch.Tensor,
    cosine_sim_yes: bool,
    batch_size: int = None,
):
    with torch.no_grad():
        n, d = x.shape
        # initial guess from low-disc sequence
        if means is None:
            # means = SobolEngine(d, seed=0).draw(k).double()
            # means = SobolEngine(d).draw(k).to(x.device, dtype=torch.float) * 2 - 1
            # means = means * 6 - 3
            means = x[torch.randperm(n)[:k], :]
        for i in range(iters):
            if not cosine_sim_yes:
                if batch_size is None:  # full-batch distance compute
                    x_means_dist = torch.cdist(x, means)
                else:
                    x_means_dist = (
                        torch.ones((n, k), dtype=x.dtype, device=x.device) * np.inf
                    )
                    for j in range(0, n, batch_size):
                        dists = torch.cdist(x[j : j + batch_size, :], means)
                        x_means_dist[j : j + batch_size, : dists.shape[-1]] = dists
            else:
                x_means_dist = 1 - torch.mm(x, means.t())
                x_means_dist /= (
                    x.norm(dim=-1, keepdim=True) * means.norm(dim=-1, keepdim=True).t()
                )
            assignments = torch.argmin(x_means_dist, dim=-1)

            # update centroids
            mask = torch.zeros((n, k), dtype=x.dtype, device=x.device)
            mask[torch.arange(n), assignments] = 1 / n
            n_assigns = mask.sum(dim=0, keepdim=True).t()
            zero_assign_yes = (n_assigns == 0).squeeze()
            zero_assign_idx = torch.where(zero_assign_yes)[0]
            n_zero_assigns = torch.sum(zero_assign_yes).cpu().numpy()
            means = torch.mm(torch.where(n_assigns == 0, 0, mask.t()), x) / torch.where(
                n_assigns == 0, 1, n_assigns
            )
            to_assign = np.minimum(n, n_zero_assigns)
            means[zero_assign_idx[:to_assign]] = x[
                torch.randperm(n)[:n_zero_assigns], :
            ]

        x_means_dist = torch.cdist(x, means)
        assignments = torch.argmin(x_means_dist, dim=-1)
    return means, assignments


def my_kmedoids(
    x: torch.Tensor,
    k: int,
    iters: int,
    medoids: torch.Tensor,
    cosine_sim_yes: bool,
    device="cuda:0",
):
    with torch.no_grad():
        n, d = x.shape
        x_x_dist = torch.cdist(x, x)
        if medoids is None:
            medoids = x[torch.randperm(n)[:k], :]

        for i in range(iters):
            if not cosine_sim_yes:
                x_medoids_dist = torch.cdist(x, medoids)
            else:
                x_medoids_dist = 1 - torch.mm(x, medoids.t())
                x_medoids_dist /= (
                    x.norm(dim=-1, keepdim=True)
                    * medoids.norm(dim=-1, keepdim=True).t()
                )

            assignments = torch.argmin(x_medoids_dist, dim=-1)

            # update centroids: within each assignment, compute medoid, which is argmin of distance between median and members
            mask = torch.zeros((n, k), dtype=torch.bool, device=x.device)
            mask[torch.arange(n), assignments] = True
            adj = torch.where(
                mask.T[..., None],
                torch.ones((n, n), dtype=torch.bool, device=device),
                False,
            )
            adj = torch.where(
                mask.T[:, None, :], adj, False
            )  # shape is (k, n, n) where adj[c] reveals adjacency matrix

            wadj = torch.where(adj, x_x_dist, torch.nan)  # shape is (k, n, n)
            medoid_idxs = torch.argmin(
                torch.where(
                    torch.any(adj, dim=-1), torch.nansum(wadj, dim=-1), torch.inf
                ),
                dim=-1,
            )

            n_assigns = mask.sum(dim=0, keepdim=True).t()
            zero_assign_yes = (n_assigns == 0).squeeze()
            n_zero_assigns = torch.sum(zero_assign_yes)

            # # medoids = torch.mm(torch.where(n_assigns == 0, 0, mask.t()), x) / torch.where(n_assigns == 0, 1, n_assigns)
            # # within each assignment, we need to choose the point that minimizes the sum of distances to all other points
            # # so each assignment must have its own distance matrix
            # # subdists = torch.ones([k, n, n], device=x.device) * torch.inf  # shape is (k, n, n)
            # subdists = torch.where(mask.T[..., None], x_x_dist, torch.inf)  # shape is (k, n, n)
            # subdists = torch.where(mask.T[:, None, :], subdists, torch.inf)
            # subdist_sum = torch.nansum(torch.where(subdists == torch.inf, torch.nan, subdists), dim=-1)  # shape is (k, n)
            # medoid_idxs = torch.argmin(subdist_sum, dim=-1)  # shape is (k,) and values range in (0, n-1)
            medoids = x[medoid_idxs]

            medoids[zero_assign_yes] = x[torch.randperm(n)[:n_zero_assigns], :]

        x_medoids_dist = torch.cdist(x, medoids)
        assignments = torch.argmin(x_medoids_dist, dim=-1)
    return medoids, assignments, medoid_idxs


class VQTokenizer(nn.Module):
    """
    To do tokenized autoregressive sampling of continuous vectors,
    we will map the high-dimensional state-goal-action tuples into token sequences.
    This is done by mapping each feature of the SGA tuple onto the non-negative integer space Z+,
    Where each feature has a fixed number of tokens $k$ in budget, unto which k-means clustering quantizes continuous values into tokens.
    """

    def __init__(self, codebook_size: int, kmeans_iters: int, cosine_sim_yes: bool):
        """
        Takes a whole dataset to build tokenization scheme from.
        Raw input data is (N, D), and we shall build D tokenization schemes,
        from which we get a non-overlapping integer embedding.

        """
        super().__init__()
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.cosine_sim_yes = cosine_sim_yes
        self.codebook = None

    def initialize(self, x: torch.Tensor, device="cuda:0"):
        """
        Use the given dataset to build the initial RVQ codebooks
        """
        self.feature_size = x.shape[1]
        x = x.to(device)
        if self.codebook is not None:
            self.codebook = self.codebook.to(device)
        self.codebook, assignments = my_kmeans(
            x, self.codebook_size, self.kmeans_iters, self.codebook, self.cosine_sim_yes
        )
        x = x.cpu()
        self.codebook = self.codebook.cpu()

    def encode(self, x: torch.Tensor, device="cuda:0"):
        if self.codebook is None or self.feature_size is None:
            raise RuntimeError("VQTokenizer not initialized")
        assert x.shape[1] == self.feature_size
        x = x.to(device)
        with torch.no_grad():
            self.codebook = self.codebook.to(device)
            raw_to_means_dist = torch.cdist(x, self.codebook)
            encoded = torch.argmin(raw_to_means_dist, dim=1)
            quantized = self.codebook[encoded]
        x = x.cpu()
        self.codebook = self.codebook.cpu()
        return encoded.detach().cpu(), quantized.detach().cpu()

    def decode(self, x: torch.Tensor):
        if self.codebook is None or self.feature_size is None:
            raise RuntimeError("VQTokenizer not initialized")
        decoded = self.codebook[x]
        return decoded


class RVQTokenizer(nn.Module):

    def __init__(
        self,
        feature_size: int,
        n_quantizers: int,
        codebook_size: int,
        kmeans_iters: int,
        kmeans_trials: int,
        cosine_sim_yes: bool,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.kmeans_trials = kmeans_trials
        self.cosine_sim_yes = cosine_sim_yes

        # self.feature_size: int or None = None
        # self.codebooks: torch.Tensor or None = None

        self.codebooks = (
            torch.rand(
                self.n_quantizers,
                self.codebook_size,
                self.feature_size,
                requires_grad=False,
                dtype=torch.float,
            )
            - 0.5
        ) * 2

    def build_codebook(
        self,
        x: torch.Tensor,
        scratch_yes: bool = True,
        batch_size: int = None,
        device="cuda:0",
    ):
        """
        Use the given dataset to build the initial RVQ codebooks
        """
        assert self.feature_size == x.shape[1]
        if scratch_yes:
            self.codebooks = torch.zeros(
                self.n_quantizers,
                self.codebook_size,
                self.feature_size,
                requires_grad=False,
                dtype=torch.float,
            )
        elif self.codebooks is None:
            raise RuntimeError(
                "scratch_yes must be False unless codebooks have been populated"
            )
        x = x.to(device)
        self.codebooks = self.codebooks.to(device)
        quantized = torch.zeros_like(x)
        for i in tqdm(range(self.n_quantizers)):
            # Hacky: support inheriting top level codebook
            if i <= 0 and not scratch_yes:
                means_, assignments = my_kmeans(
                    x - quantized,
                    self.codebook_size,
                    self.kmeans_iters,
                    self.codebooks[i],
                    self.cosine_sim_yes,
                    batch_size,
                )
                self.codebooks[i] = means_
                # x_means_dist = torch.cdist(x, self.codebooks[i])
                # assignments = torch.argmin(x_means_dist, dim=-1)
            else:
                means_, assignments = my_kmeans(
                    x - quantized,
                    self.codebook_size,
                    self.kmeans_iters,
                    None,
                    self.cosine_sim_yes,
                    batch_size,
                )
                self.codebooks[i] = means_
            quantized += self.codebooks[i][assignments]
        x = x.cpu()
        self.codebooks = self.codebooks.cpu()

    def encode(self, x: torch.Tensor, device="cuda:0"):
        if self.codebooks is None or self.feature_size is None:
            raise RuntimeError("RVQTokenizer not initialized")
        assert x.shape[1] == self.feature_size
        x = x.to(device)
        self.codebooks = self.codebooks.to(device)
        encoded = torch.zeros(
            x.shape[0],
            self.n_quantizers,
            dtype=torch.long,
            device=x.device,
            requires_grad=False,
        )
        quantized = torch.zeros_like(x, requires_grad=False)
        for j in range(self.n_quantizers):
            raw_to_means_dist = torch.cdist(x - quantized, self.codebooks[j])
            encoded[:, j] = torch.argmin(raw_to_means_dist, dim=1)
            quantized += self.codebooks[j, encoded[:, j]]
        x = x.cpu()
        self.codebooks = self.codebooks.cpu()
        return encoded, quantized

    def decode(self, x: torch.Tensor):
        if self.codebooks is None or self.feature_size is None:
            raise RuntimeError("RVQTokenizer not initialized")
        assert x.shape[1] == self.n_quantizers
        gathered = torch.gather(
            self.codebooks.swapaxes(0, 1),
            0,
            index=x.unsqueeze(-1).expand(-1, -1, self.feature_size),
        )
        decoded = torch.sum(gathered, dim=1)
        return decoded


class BucketizeTokenizer(nn.Module):
    """
    Baseline, but probably good enough
    """

    def __init__(
        self,
        codebook_size: int,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebooks = torch.as_tensor(
            np.mgrid[min_value : max_value : complex(0, codebook_size)],
            dtype=torch.float,
        )
        self.min_value = min_value
        self.max_value = max_value
        assert self.codebook_size > 0

    def initialize(self, x: torch.Tensor, device="cuda:0"):
        pass

    def encode(self, x: torch.Tensor, device="cuda:0"):
        # encoded = torch.argmin(torch.abs(x.unsqueeze(-1) - self.codebooks), dim=-1)
        # quantized = self.codebooks[encoded]

        encoded = torch.clip(x, self.min_value, self.max_value)  # [min, max]
        encoded -= self.min_value  # [0, max - min]
        encoded /= self.max_value - self.min_value  # [0, 1]
        encoded *= float(self.codebook_size - 1)
        # encoded = encoded.long().to(device)
        encoded = torch.round(encoded).long().to(device)

        quantized = self.decode(encoded, device=device)

        return encoded, quantized

    def decode(self, x: torch.Tensor, device="cuda:0"):
        decoded = x.float() / (self.codebook_size - 1)
        decoded *= self.max_value - self.min_value
        decoded += self.min_value
        #
        # decoded = self.codebooks[x]
        return decoded.to(device)
