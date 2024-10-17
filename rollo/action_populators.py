import numpy as np


class ActionPopulator:
    def __init__(self):
        pass

    def populate(self, action):
        return action


class BilliardsActionPopulator(ActionPopulator):

    def __init__(self):
        pass

    def populate(self, action):
        a = action * np.pi
        a = 1.0 * np.concatenate([np.cos(a), np.sin(a)], axis=-1)
        return a
