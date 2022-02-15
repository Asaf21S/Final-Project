from MatchNet import MatchNet
import numpy as np


class FastMatch:
    def __init__(self, epsilon=0.15, delta=0.25, photometric_invariance=False, min_scale=0.5, max_scale=2):
        self.epsilon = epsilon
        self.delta = delta
        self.photometric_invariance = photometric_invariance
        self.min_scale = min_scale
        self.max_scale = max_scale

    def run(self, image, template):
        # implementation

        # just some functionality testing
        net = MatchNet(1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)
        print(net)
        print(net * 3)
        print(net / 5)

        corners = [(20, 30), (100, 35), (120, 90), (25, 100)]
        return np.array([corners])
