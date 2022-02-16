from MatchNet import MatchNet
from MatchConfig import MatchConfig
import numpy as np
import cv2
from random import randrange
'''
example of multiprocessing (for CPU-bound usage):
https://www.analyticsvidhya.com/blog/2021/04/a-beginners-guide-to-multi-processing-in-python/#:~:text=regular%20for%20loop.-,Using%20a%20Pool%20class%2D,-import%20time%0Aimport
from multiprocessing import Pool
'''

# import matplotlib.pyplot as plt


class FastMatch:
    def __init__(self, epsilon=0.15, delta=0.25, photometric_invariance=False, min_scale=0.5, max_scale=2):
        self.epsilon = epsilon
        self.delta = delta
        self.photometric_invariance = photometric_invariance
        self.min_scale = min_scale
        self.max_scale = max_scale

    def run(self, image, template):
        # preprocess the images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        template = cv2.normalize(template, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if image.shape[0] % 2 == 0:
            image = image[0:image.shape[0] - 1, 0:image.shape[1]]
        if image.shape[1] % 2 == 0:
            image = image[0:image.shape[0], 0:image.shape[1] - 1]
        if template.shape[0] % 2 == 0:
            template = template[0:template.shape[0] - 1, 0:template.shape[1]]
        if template.shape[1] % 2 == 0:
            template = template[0:template.shape[0], 0:template.shape[1] - 1]

        # create the matching net
        min_tx = 0.5 * ((template.shape[0] - 1) * self.min_scale - (image.shape[0] - 1))
        min_ty = 0.5 * ((template.shape[1] - 1) * self.min_scale - (image.shape[1] - 1))
        net = MatchNet(template.shape[0], template.shape[1], self.delta,
                       min_tx, -min_tx, min_ty, -min_ty, -np.pi, np.pi, self.min_scale, self.max_scale)

        # smooth the images
        image = cv2.GaussianBlur(image, (0, 0), 2, 2)
        template = cv2.GaussianBlur(template, (0, 0), 2, 2)

        # randomly sample points
        samples = [(randrange(1, template.shape[0]), randrange(1, template.shape[1])) for _ in range(np.rint(
            10 / np.square(self.epsilon)).astype(np.int32))]

        level = 0
        delta_fact = 1.511
        new_delta = self.delta
        best_config = MatchConfig()
        best_trans = np.empty([3, 3], dtype=float)
        best_distances = np.zeros(20)  # might need to change to python list instead of numpy array if resizing required
        best_distance = 0.0
        distances = []
        insiders = []

        while True:
            level += 1

            # first create configurations based on our net
            configs, affines = self.create_list_of_configs(net, template.shape, image.shape)

            # calculate distance for each configuration
            distances = self.evaluate_configs()
            break

        corners = [(20, 30), (100, 35), (120, 90), (25, 100)]
        return np.array([corners])

    @staticmethod
    def create_list_of_configs(net: MatchNet, template_size, image_size):
        # creating the steps for all the parameters
        tx_steps = net.get_tx_steps()
        ty_steps = net.get_ty_steps()
        r_steps = net.get_rotation_steps()
        s_steps = net.get_scale_steps()

        r2_steps_amount = len(r_steps)
        if np.abs(net.rot_bounds[1] - net.rot_bounds[0]) < 2 * np.pi + 0.1:
            r2_steps_amount = sum(1 for r in r_steps if r < (net.rot_steps - np.pi) / 2)

        # iterate through each possible affine configuration steps
        configs = np.empty((len(tx_steps), len(ty_steps), r2_steps_amount, len(s_steps), len(s_steps), len(r_steps)), dtype=object)
        affines = np.empty(configs.shape + (2, 3))
        for indices, _ in np.ndenumerate(configs):
            match_config = MatchConfig(indices)
            configs[indices] = match_config
            affines[indices] = match_config.get_affine_matrix()

        # filter configurations which fall outside of image boundaries
        corners = np.ones((3, 4))
        corners[0, 1] = corners[0, 2] = template_size[0]
        corners[1, 2] = corners[1, 3] = template_size[1]
        corners[0] -= template_size[0] / 2 + 0.5
        corners[1] -= template_size[1] / 2 + 0.5

        affine_corners = np.matmul(affines, corners)
        valid_configs = []
        valid_affines = []
        for indices, _ in np.ndenumerate(configs):
            affine_corners[indices][0] += image_size[0] / 2 + 0.5
            affine_corners[indices][1] += image_size[1] / 2 + 0.5
            corners_X = affine_corners[indices][0]
            corners_Y = affine_corners[indices][1]
            if all(-10 < x < image_size[0] + 10 for x in corners_X) and all(-10 < y < image_size[1] + 10 for y in corners_Y):
                valid_configs.append(configs[indices])
                valid_affines.append(affines[indices])

        return valid_configs, valid_affines

    def evaluate_configs(self):
        return []

    '''
# for debugging
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("image")
plt.subplot(1, 2, 2)
plt.imshow(template, cmap='gray')
plt.title("template")
plt.show()
'''
