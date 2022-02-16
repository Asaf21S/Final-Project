import numpy as np


class MatchConfig:
    def __init__(self, *args):
        if len(args) == 1:
            self.init(args[0][0], args[0][1], args[0][2], args[0][3], args[0][4], args[0][5])

    def init(self, trans_x, trans_y, rotate_2, scale_x, scale_y, rotate_1):
        # ...
        pass

    def get_affine_matrix(self):
        # ...
        affine = np.zeros((2, 3))
        return affine
