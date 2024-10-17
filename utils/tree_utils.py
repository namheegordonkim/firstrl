import jax
import jax.numpy as jp


def tree_concatenate(trees, axis=0):
    """Takes a list of trees and concatenates every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((concatenate(a, a'), concatenate(b, b')), concatenate(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = []
    for l in grouped_leaves:
        if isinstance(l[0], jp.ndarray):
            result_leaves.append(jp.concatenate(l, axis=axis))
        else:
            result_leaves.append(tree_concatenate(l, axis=axis))
    return jax.tree_util.tree_unflatten(treedef_list[0], result_leaves)


def tree_stack(trees, axis=0):
    """Takes a list of trees and concatenates every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((concatenate(a, a'), concatenate(b, b')), concatenate(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = []
    for l in grouped_leaves:
        result_leaves.append(jp.stack(l, axis=axis))
    return jax.tree_util.tree_unflatten(treedef_list[0], result_leaves)
