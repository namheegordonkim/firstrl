import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from utils.torch_nets import GCFitPredict
from utils.train_utils import RunningMeanStd


class MyKNeighborsRegressor:
    def __init__(self, n_neighbors: int, weights: str = "uniform"):
        self.model = KNeighborsRegressor(n_neighbors, weights=weights)
        self.n_neighbors = n_neighbors
        self.a_tokenizer = None
        self.gh_tokenizer = None
        self.nail_scaler = None

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.args = args
        instance.kwargs = kwargs
        return instance

    def fit(self, X, W, y):
        self.X = X
        self.W = W
        self.y = y
        fit_in = np.concatenate([X, W], axis=-1)
        self.model.fit(fit_in, y)

    def predict(self, X: np.ndarray, W: np.ndarray):
        pred_in = np.concatenate([X, W], axis=-1)
        return self.model.predict(pred_in)


class FilteringNeighbors:

    def __init__(self, n_proposals: int, obs_size: int, goal_size: int):
        self.x_0_dots_T = None
        self.m_1_dots_T = None
        self.max_horizon = None
        self.xp_xpos = None

        self.rms_o0 = None
        self.rms_g = None

        self.obs_size = obs_size
        self.goal_size = goal_size

        self.rms_o0 = RunningMeanStd(shape=(self.obs_size,))
        self.rms_g = RunningMeanStd(shape=(self.goal_size,))
        self.n_proposals = n_proposals

    def fit(self, x0_dots_T, m_1_dots_T):
        self.x_0_dots_T = x0_dots_T
        self.m_1_dots_T = m_1_dots_T
        self.max_horizon = x0_dots_T.shape[1] - 1
        self.xp_xpos = self.x_0_dots_T
        # self.xp_xpos = self.xp_xpos.reshape((*self.xp_xpos.shape[:-2], -1))

        self.rms_o0.update(self.xp_xpos[..., :-1].reshape(-1, self.obs_size))
        self.rms_g.update(self.xp_xpos[..., -1, :-1].reshape(-1, self.goal_size))

    def predict(self, o0: np.ndarray, g: np.ndarray):
        max_horizon = self.max_horizon

        # Want to compute two distance matrices comparing o0 to xp and g to xp
        left = self.xp_xpos[None, :, 0, ..., :-1]
        left = left.reshape(*left.shape[:-2], self.obs_size)
        right = o0[:, None]
        d1 = np.linalg.norm(
            self.rms_o0.normalize(left) - self.rms_o0.normalize(right),
            axis=-1,
        )

        left = self.xp_xpos[None, ..., [-1], :-1]
        left = left.reshape(*left.shape[:-2], self.goal_size)
        right = g[:, None, None]
        d2 = np.linalg.norm(
            self.rms_g.normalize(left) - self.rms_g.normalize(right),
            axis=-1,
        )
        dd2 = d2.min(-1)

        if self.n_proposals == 1:
            blend_weights = np.linspace(0.5, 0.5, 1, endpoint=True)
        else:
            blend_weights = np.linspace(0, 1, self.n_proposals, endpoint=True)
        blend_weights = blend_weights[:, None, None]
        d = d1[None] * blend_weights + dd2[None] * (1 - blend_weights)

        # d = d1 + dd2
        i = d.argmin(-1)
        # j = d.min(-2).argmin(-1)
        a = self.m_1_dots_T[i]

        return a


class MyKNeighbors(GCFitPredict):

    def __init__(self, n_proposals: int, obs_size: int, goal_size: int):
        self.o0 = None
        self.g = None
        self.m_1_dots_T = None

        self.max_horizon = None
        self.xp_o0 = None
        self.xp_g = None

        self.rms_o0 = None
        self.rms_g = None

        self.obs_size = obs_size
        self.goal_size = goal_size

        self.rms_o0 = RunningMeanStd(shape=(self.obs_size,))
        self.rms_g = RunningMeanStd(shape=(self.goal_size,))
        self.n_proposals = n_proposals

    def fit(self, o0, g, m_1_dots_T):
        self.o0 = o0
        self.g = g
        self.m_1_dots_T = m_1_dots_T
        self.max_horizon = o0.shape[1] - 1

        self.rms_o0.update(o0)
        self.rms_g.update(g)

    def predict(self, o0: np.ndarray, g: np.ndarray):
        max_horizon = self.max_horizon

        # Want to compute two distance matrices comparing o0 to xp and g to xp
        left = self.o0[None]
        right = o0[:, None]
        d1 = np.linalg.norm(
            self.rms_o0.normalize(left) - self.rms_o0.normalize(right),
            # left - right,
            axis=-1,
        )

        left = self.g[None]
        right = g[:, None]
        d2 = np.linalg.norm(
            self.rms_g.normalize(left) - self.rms_g.normalize(right),
            # left - right,
            axis=-1,
        )

        if self.n_proposals == 1:
            d = d1[None] + d2[None]
        else:
            blend_weights = np.linspace(0, 1, self.n_proposals, endpoint=True)
            blend_weights = blend_weights[:, None, None]
            d = d1[None] * blend_weights + d2[None] * (1 - blend_weights)

        # d = d1 + dd2
        i = d.argmin(-1)
        # j = d.min(-2).argmin(-1)
        a = self.m_1_dots_T[i]

        return a
