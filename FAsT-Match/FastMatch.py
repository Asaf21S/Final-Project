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
        samples_loc = [(randrange(1, template.shape[0]), randrange(1, template.shape[1])) for _ in range(np.rint(
            10 / np.square(self.epsilon)).astype(np.int32))]

        level = 0
        delta_fact = 1.511
        new_delta = self.delta
        configs = []
        affines = []
        best_config = MatchConfig()
        best_affine = np.zeros((2, 3))
        best_distances = np.zeros(20)  # might need to change to python list instead of numpy array if resizing required
        best_distance = 0.0
        distances = []
        insiders = []

        while True:
            level += 1

            # first create configurations based on our net
            if level == 1:
                configs, affines = self.create_list_of_configs(net, template.shape, image.shape)

            # calculate distance for each configuration
            distances = self.evaluate_configs(image, template, affines, samples_loc)

            # find the minimum distance
            best_distance = best_distances[level] = min(distances)
            min_index = distances.index(best_distance)
            best_config = configs[min_index]
            best_affine = affines[min_index]

            # conditions to exit the loop
            if best_distance < 0.005 or (level > 2 and best_distance < 0.015) or level >= 20:
                break
            if level > 3 and best_distance > ((best_distances[level - 3] + best_distances[level - 2] +
                                               best_distances[level - 1]) / len(distances)) * 0.97:
                break

            # get the good configurations that falls between certain thresholds
            good_configs, good_affines = self.get_good_configs(configs, affines, best_distance, new_delta, distances)
            if level == 1 and ((len(good_configs) / len(configs) > 0.022 and best_distance > 0.05 and
                                len(configs) < 7.5e6) or (best_distance > 0.1 and len(configs) < 5e6)):
                # restart
                factor = 0.9
                new_delta *= factor
                level -= 1
                net *= factor
            else:
                new_delta /= delta_fact
                number_of_points = 80  # amount of new points for each good configuration
                expanded_configs_mat = self.random_expand_configs(good_configs, net, new_delta, number_of_points)
                # expanded_config_mat is (1, 6) representation of configuration. this is NOT the affine matrix
                expanded_configs, expanded_affines = self.get_list_of_configs(expanded_configs_mat, template.shape, image.shape)

                configs = [good_configs, expanded_configs]
                affines = [good_affines, expanded_affines]
                # or something like that
                # make sure that configs and affines are the same variables as before and not local variables
                # add comments on everything

            '''/* Randomly sample points again */
            rng.fill( xs, RNG::UNIFORM, 1, templ.cols );
            rng.fill( ys, RNG::UNIFORM, 1, templ.rows );'''

        '''/* Return the rectangle corners based on the best affine transformation */
        return calcCorners( image.size(), templ.size(), best_trans );'''
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

        # print(corners)
        affine_corners = np.matmul(affines, corners)
        # print(affine_corners[0])
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

    def evaluate_configs(self, image, template, affines, samples_loc):
        amount_of_points = len(samples_loc)

        padded_image = np.pad(image, ((image.shape[1], image.shape[1]), (0, 0)))
        template_samples = np.array([template[pnt[1] - 1, pnt[0] - 1] for pnt in samples_loc])
        samples_loc = np.array(samples_loc).transpose()
        samples_loc[0] -= int((template.shape[0] + 1) / 2)
        samples_loc[1] -= int((template.shape[1] + 1) / 2)
        samples_loc = np.vstack([samples_loc, [1.0] * amount_of_points])
        epsilon = 1.0e-7

        # calculate the score for each configuration on each of our randomly samples point
        # maybe use parallel processing
        '''
        pool = Pool()  # create a pool of processes
        pool.map(self.calc_dist, range(len(affines)))  # call calc_dist for each affine matrix
        pool.close()  # prevent any more tasks from being submitted to the pool
        '''
        distances = []
        for affine in affines:
            affine[0, 2] += image.shape[0] / 2 + 1
            affine[1, 2] += image.shape[1] / 2 + 1 + image.shape[1]
            transformed_samples_loc = np.matmul(affine, samples_loc)
            transformed_samples_loc = transformed_samples_loc.astype(int) - 1
            transformed_samples_loc = [(transformed_samples_loc[0, i], transformed_samples_loc[1, i]) for i in range(amount_of_points)]
            image_samples = np.array([padded_image[pnt[1] - 1, pnt[0] - 1] for pnt in transformed_samples_loc])

            if not self.photometric_invariance:
                score = np.sum(np.abs(template_samples - image_samples))
            else:
                sums = (np.sum(template_samples), np.sum(image_samples),
                        np.sum(np.square(template_samples)), np.sum(np.square(image_samples)))
                sigma_x = np.sqrt((sums[2] - np.square(sums[0]) / amount_of_points) / amount_of_points) + epsilon
                sigma_y = np.sqrt((sums[3] - np.square(sums[1]) / amount_of_points) / amount_of_points) + epsilon
                sigma_div = sigma_x / sigma_y
                score = np.sum(np.abs(template_samples - (image_samples * sigma_div) + (sums[1] * sigma_div + sums[0]) / amount_of_points))

            distances.append(score / amount_of_points)
        return distances

    def get_good_configs(self, configs, affines, best_distance, new_delta, distances):
        threshold = best_distance + self.get_threshold(new_delta)
        good_configs = []
        good_affines = []
        for i in range(len(configs)):
            if distances[i] <= threshold:
                good_configs.append(configs[i])
                good_affines.append(affines[i])
        while len(good_configs) > 27000:
            good_configs = []
            good_affines = []
            threshold *= 0.99
            for i in range(len(configs)):
                if distances[i] <= threshold:
                    good_configs.append(configs[i])
                    good_affines.append(affines[i])
        if len(good_configs) == 0:
            print("Error: no good configurations were found")
        return good_configs

    @staticmethod
    def get_threshold(delta):
        p0 = 0.1341
        p1 = 0.0278
        safety = 0.02
        return p0 * delta + p1 - safety

    def random_expand_configs(self, configs, net, factor, number_of_points):
        # create random 2x3 matrices
        random_vec = np.random.normal(0, 0.5, size=(number_of_points * len(configs), 2, 3))  # need to be changed to size=(number_of_points * len(configs), 6)
        # NEED TO ADD: random_vec.convertTo( random_vec, CV_32FC1 );
        ranges = np.empty((1, 2, 3))  # need to be changed to (1, 6)
        ranges[0][0][0] = net.tx_steps / factor  # need to be changed accordingly
        ranges[0][0][1] = net.ty_steps / factor  # need to be changed accordingly
        ranges[0][0][2] = ranges[0][1][2] = net.rot_steps / factor  # need to be changed accordingly
        ranges[0][1][0] = ranges[0][1][1] = net.sc_steps / factor  # need to be changed accordingly
        ranges = np.repeat(ranges, number_of_points * len(configs), axis=0)  # don't change
        # change this to (1, 6) representation of configuration. this is NOT the affine matrix
        expanded = np.array(configs.asMatrix)  # (123120, 2, 3) -> need to be changed to (123120, 6)
        expanded = np.repeat(expanded, number_of_points, axis=0)  # don't change
        expanded_config_mat = expanded + random_vec * ranges  # don't change
        print(expanded_config_mat.shape)
        return expanded_config_mat

    def get_list_of_configs(self, config_mat, template_size, image_size):
        # config_mat.shape = (amount_of_good_configs * expension_factor, 2, 3) -> need to be changed to (..., 6)
        # this function need to do the same like create_list_of_configs but using the config_mat instead of Net
        # that mean check for boundries too. and return list of configs and list of affines
        return [], []

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
