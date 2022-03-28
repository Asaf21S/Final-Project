import numpy as np
import copy


class MatchNet:
    def __init__(self, width: int, height: int, delta: float,
                 min_tx: float, max_tx: float, min_ty: float, max_ty: float,
                 min_rot: float, max_rot: float, min_sc: float, max_sc: float):
        if not 0 <= min_sc <= 1:
            raise Exception("min_sc should be between 0 and 1")
        if not 1 <= max_sc <= 5:
            raise Exception("max_sc should be between 1 and 5")
        if not -np.pi <= min_rot <= 0:
            raise Exception("min_rot should be between -pi and 0")
        if not 0 <= max_rot <= np.pi:
            raise Exception("max_rot should be between 0 and pi")

        self.tx_bounds = (min_tx, max_tx)
        self.ty_bounds = (min_ty, max_ty)
        self.rot_bounds = (min_rot, max_rot)
        self.sc_bounds = (min_sc, max_sc)

        self.tx_steps = delta * width / np.sqrt(2)
        self.ty_steps = delta * height / np.sqrt(2)
        self.rot_steps = delta * np.sqrt(2)
        self.sc_steps = delta / np.sqrt(2)

    def get_tx_steps(self):
        # 'pad' at end of range with an extra sample
        return np.arange(self.tx_bounds[0], self.tx_bounds[1] + 0.5 * self.tx_steps, self.tx_steps)

    def get_ty_steps(self):
        # 'pad' at end of range with an extra sample
        return np.arange(self.ty_bounds[0], self.ty_bounds[1] + 0.5 * self.ty_steps, self.ty_steps)

    def get_rotation_steps(self):
        # np.arange(self.rot_bounds[0], self.rot_bounds[1] + 0.5 * self.rot_steps, self.rot_steps)
        # Rotations ignore the user selected range here - it is handled in main function
        # no padding since it is a cyclic range
        return np.arange(-np.pi, np.pi, self.rot_steps)

    def get_scale_steps(self):
        if self.sc_steps == 0.0:
            return np.array([self.sc_bounds[0]])
        # 'pad' at end of range with an extra sample
        return np.arange(self.sc_bounds[0], self.sc_bounds[1] + 0.5 * self.sc_steps, self.sc_steps)

    def __mul__(self, factor):
        if isinstance(factor, (int, float)):
            mn = copy.deepcopy(self)
            mn.tx_steps *= factor
            mn.ty_steps *= factor
            mn.rot_steps *= factor
            mn.sc_steps *= factor
            return mn

    def __truediv__(self, factor):
        if isinstance(factor, (int, float)):
            mn = copy.deepcopy(self)
            mn.tx_steps /= factor
            mn.ty_steps /= factor
            mn.rot_steps /= factor
            mn.sc_steps /= factor
            return mn

    def __str__(self):
        return "(\n\ttx_bounds: {0}\n\tty_bounds: {1}\n\trot_bounds: {2}\n\tsc_bounds: {3}\n\ttx_steps: {4}\n\t" \
               "ty_steps: {5}\n\trot_steps: {6}\n\tsc_steps: {7}\n)".format(self.tx_bounds, self.ty_bounds,
                                                                            self.rot_bounds, self.sc_bounds,
                                                                            self.tx_steps, self.ty_steps,
                                                                            self.rot_steps, self.sc_steps)
