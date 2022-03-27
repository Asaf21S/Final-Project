from MatchNet import MatchNet
import numpy as np
import cv2
from random import randrange
import time

'''
example of multiprocessing (for CPU-bound usage):
https://www.analyticsvidhya.com/blog/2021/04/a-beginners-guide-to-multi-processing-in-python/#:~:text=regular%20for%20loop.-,Using%20a%20Pool%20class%2D,-import%20time%0Aimport
from multiprocessing import Pool
'''

class FastMatch:
    def __init__(self, epsilon=0.15, delta=0.25, photometric_invariance=False, min_scale=0.5, max_scale=2):
        self.epsilon = epsilon
        self.delta = delta
        self.photometric_invariance = photometric_invariance
        self.min_scale = min_scale
        self.max_scale = max_scale
        # add template_mask

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

        print("image.shape", image.shape)
        print("template.shape", template.shape)
        # parametrize the initial grid
        min_tx = 0.5 * ((template.shape[0] - 1) * self.min_scale - (image.shape[0] - 1))
        min_ty = 0.5 * ((template.shape[1] - 1) * self.min_scale - (image.shape[1] - 1))
        net = MatchNet(template.shape[0], template.shape[1], self.delta,
                       min_tx, -min_tx, min_ty, -min_ty, -np.pi, np.pi, self.min_scale, self.max_scale)

        # generate Theta(1/eps^2) random points (and fresh ones each iteration later on)
        num_points = np.round(10 / np.square(self.epsilon))
        samples_loc = [(randrange(1, template.shape[0]), randrange(1, template.shape[1])) for _ in range(np.rint(
            num_points).astype(np.int32))]

        # generate the Net
        ## configs, affines = self.create_list_of_configs(net, template.shape, image.shape)
        configs = self.create_list_of_configs(net)

        if configs.shape[0] > 71000000:
            raise Exception("more than 35 million configs!")

        level = 0
        delta_fact = 1.511
        new_delta = self.delta
        best_dists = np.empty(20)
        num_configs = np.empty(20)
        num_good_configs = np.empty(20)
        orig_percentages = np.empty(20)
        total_time = 0
        best_config = np.empty(6)
        best_affine = np.empty((2, 3))

        ## best_distances = np.zeros(20)  # might need to change to python list instead of numpy array if resizing required
        ## best_distance = 0.0
        ## distances = []
        ## insiders = []

        while True:
            level += 1

            # slightly blur to reduce total-variation
            blur_sigma = 1.5 + 0.5 / np.power(delta_fact, level - 1)
            image = cv2.GaussianBlur(image, (0, 0), blur_sigma, blur_sigma)
            template = cv2.GaussianBlur(template, (0, 0), blur_sigma, blur_sigma)

            # 0] if limited rotation range - filter out illegal rotations
            if net.rot_bounds[0] > -np.pi or net.rot_bounds[1] < np.pi:
                # total rotation in the range [0,2*pi]
                total_rots = configs[:, 2] + configs[:, 5] % (2 * np.pi)
                # total rotation in the range [-pi,pi]
                total_rots[total_rots > np.pi] -= 2 * np.pi
                # filtering
                configs = configs[net.rot_bounds[0] <= total_rots <= net.rot_bounds[1], :]

            # 1] translate config vectors to matrix form
            tic = time.time()
            print("----- Configs2Affine, with {} configs -----".format(configs.shape[0]))
            configs, affines = self.configs_to_affines(configs, image.shape, template.shape)
            print('Done Configs2Affine in {:.8f} seconds'.format(time.time() - tic))
            #affines = 0

            # 2] evaluate all configurations
            tic = time.time()
            print("----- evaluate_configs, with {} configs -----".format(configs.shape[0]))
            distances = self.evaluate_configs(image, template, affines, samples_loc)
            print('Done Evaluation in {:.8f} seconds'.format(time.time() - tic))
            # find the minimum distance
            best_distance = min(distances)
            best_dists[level] = best_distance
            min_index = distances.index(best_distance)
            best_config = configs[min_index]
            best_affine = affines[min_index]
            print("bestDist = {:.3}".format(best_distance))

            # 3] choose the 'surviving' configs and delta for next round
            tic = time.time()
            print("----- Getting good configs, with {} configs -----".format(configs.shape[0]))
            good_configs, good_affines = self.get_good_configs(configs, affines, best_distance, new_delta, distances)
            orig_percentage = good_configs.shape[0] / configs.shape[0]
            print("3\tpercentage of good configs = {:.3}".format(orig_percentage))

            # collect round stats
            num_configs[level] = configs.shape[0]
            num_good_configs[level] = good_configs.shape[0]
            orig_percentages[level] = orig_percentage

            print('Done Getting good configs in {:.8f} seconds'.format(time.time() - tic))

            # 4] break conditions of Branch-and-Bound
            tic = time.time()
            conditions = np.zeros(6, dtype=bool)
            conditions[0] = best_distance < 0.005  # good enough 1
            conditions[1] = level > 5 and best_distance < 0.01  # good enough 2
            conditions[2] = level >= 20  # enough levels
            conditions[3] = level > 3 and best_distance > ((best_dists[level - 3] + best_dists[level - 2] + best_dists[level - 1]) / 3) * 0.97  # no improvement in last 3 rounds
            conditions[4] = level > 2 and good_configs.shape[0] > 1000 and orig_percentage > 0.2  # too high expansion rate
            conditions[5] = level > 3 and good_configs.shape[0] > 1000 and good_configs.shape[0] > 50 * np.min(num_good_configs[:level])  # a deterioration in the focus

            if any(conditions):
                print("breaking BnB at level {}. conditions: {}".format(level, conditions))
                print("best distances by round:  {}".format(best_dists))
                print("num configs per round:    {}".format(num_configs))
                print("# good configs per round: {}".format(num_good_configs))
                print("percentage to expand:     {}".format(orig_percentages))
                break

            # 5] expand 'surviving' configs for next round
            if level == 1 and ((orig_percentage > 0.05 and best_distance > 0.1 and configs.shape[0] < 7.5e6) or
                               (orig_percentage >= 0.01 and best_distance > 0.15 and configs.shape[0] < 5e6)):
                factor = 0.9
                print("##### RESTARTING!!! changing from delta: {:.3f}, to delta: {:.3f}".format(new_delta, new_delta * factor))
                new_delta *= factor
                level = 0
                net *= factor
                configs = self.create_list_of_configs(net)
            else:
                print("##### CONTINUING!!! prevDelta = {:.3f},  newDelta = {:.3f}".format(new_delta,
                                                                                                 new_delta / delta_fact))
                new_delta /= delta_fact
                # expand the good configs
                number_of_points = 80  # amount of new points for each good configuration
                expanded_configs = self.random_expand_configs(good_configs, net, level, number_of_points, delta_fact)
                configs = np.concatenate((good_configs, expanded_configs))

            print("\n*** level {}:|goodConfigs| = {}, |expandedConfigs| = {}\n", level, good_configs.shape[0], configs.shape[0])

            # 6] refresh random points
            samples_loc = [(randrange(1, template.shape[0]), randrange(1, template.shape[1])) for _ in range(np.rint(
            num_points).astype(np.int32))]

        return self.get_corners(template.shape, best_affine)

    @staticmethod
    def create_list_of_configs(net: MatchNet):  ## , template_size, image_size):
        print("in create_list_of_configs()")
        # creating the steps for all the parameters
        tx_steps = net.get_tx_steps()
        ty_steps = net.get_ty_steps()
        r_steps = net.get_rotation_steps()
        s_steps = net.get_scale_steps()

        ntx_steps = len(tx_steps)
        nty_steps = len(ty_steps)
        ns_steps = len(s_steps)
        nr_steps = len(r_steps)

        # second rotation is a special case (can be limited to a single quartile)
        quartile1_r_steps = r_steps[r_steps < (net.rot_steps - np.pi) / 2]
        nr2_steps = len(quartile1_r_steps)

        # gridsize
        grid_size = ntx_steps * nty_steps * np.square(ns_steps) * nr_steps * nr2_steps

        # iterate through each possible affine configuration steps
        configs = np.empty((grid_size, 6))
        ## affines = np.empty(configs.shape + (2, 3))

        ## trigs = MatchConfig.get_trigs(len(r_steps), r2_steps_amount)
        ## scales = MatchConfig.get_scales(len(s_steps), trigs)

        # pool = Pool()
        # pool.starmap(FastMatch.initialize_configs, [(self, indices) for indices, _ in np.ndenumerate(configs)])
        # pool.close()

        tx, ty, r1, r2, sx, sy = 0, 0, 0, 0, 0, 0
        grid_ind = 0

        tic = time.time()
        print("Start to initialize every configuration")
        for tx_ind in range(ntx_steps):
            tx = tx_steps[tx_ind]
            print("step {} out of {}".format(tx_ind + 1, ntx_steps))
            for ty_ind in range(nty_steps):
                ty = ty_steps[ty_ind]
                for r1_ind in range(nr_steps):
                    r1 = r_steps[r1_ind]
                    for r2_ind in range(nr2_steps):
                        r2 = r_steps[r2_ind]  # maybe should be quartile1_r_steps instead of r_steps
                        for sx_ind in range(ns_steps):
                            sx = s_steps[sx_ind]
                            for sy_ind in range(ns_steps):
                                sy = s_steps[sy_ind]

                                configs[grid_ind] = np.array([tx, ty, r2, sx, sy, r1])
                                grid_ind += 1

        ## for indices, _ in np.ndenumerate(configs):
        ##     match_config = MatchConfig(indices)
        ##     configs[indices] = match_config
        ##     affines[indices] = match_config.get_affine_matrix(scales[indices[2]][indices[3]][indices[4]][indices[5]])
        print('Done in {:.8f} seconds'.format(time.time() - tic))
        return configs

    @staticmethod
    def configs_to_affines(configs, im_shape, te_shape):
        affines = np.empty((configs.shape[0], 2, 3))
        current_config = 0
        affine_mat = np.empty((2, 3))
        c1c2, s1s2, c1s2, s1c2 = 0, 0, 0, 0
        for i in range(configs.shape[0]):
            current_config = configs[i]
            c1c2 = np.cos(current_config[5]) * np.cos(current_config[2])
            s1s2 = np.sin(current_config[5]) * np.sin(current_config[2])
            c1s2 = np.cos(current_config[5]) * np.sin(current_config[2])
            s1c2 = np.sin(current_config[5]) * np.cos(current_config[2])

            affine_mat[0, 0] = current_config[3] * c1c2 - current_config[4] * s1s2
            affine_mat[0, 1] = -current_config[3] * c1s2 - current_config[4] * s1c2
            affine_mat[0, 2] = current_config[0]
            affine_mat[1, 0] = current_config[3] * s1c2 + current_config[4] * c1s2
            affine_mat[1, 1] = current_config[4] * c1c2 - current_config[3] * s1s2
            affine_mat[1, 2] = current_config[1]

            affines[i] = affine_mat

        # filter configurations which fall outside of image boundaries
        corners = np.ones((3, 4))
        corners[0, 1] = corners[0, 2] = te_shape[0]
        corners[1, 2] = corners[1, 3] = te_shape[1]
        corners[0] -= te_shape[0] / 2 + 0.5
        corners[1] -= te_shape[1] / 2 + 0.5

        tic = time.time()
        affine_corners = np.matmul(affines, corners)
        affine_corners[:, 0] += im_shape[0] / 2 + 0.5
        affine_corners[:, 1] += im_shape[1] / 2 + 0.5

        print("1\tchecking the transformed corners")
        tic = time.time()
        insiders = np.zeros(configs.shape[0])
        for i in range(configs.shape[0]):
            # if all(-10 < x < im_shape[0] + 10 for x in affine_corners[i, 0]) and all(
                    # -10 < y < im_shape[1] + 10 for y in affine_corners[i, 1]):
            if all(0 < x < im_shape[0] for x in affine_corners[i, 0]) and all(
                    0 < y < im_shape[1] for y in affine_corners[i, 1]):
                insiders[i] = 1

        configs = configs[insiders == 1, :]
        affines = affines[insiders == 1, :, :]
        print('1\tDone checking the transformed corners in {:.8f} seconds'.format(time.time() - tic))
        print("1\treduced amount of configs to", configs.shape[0])

        return configs, affines

    def evaluate_configs(self, image, template, affines, samples_loc):
        print("2\tin evaluate_configs()")
        amount_of_points = len(samples_loc)

        padded_image = np.pad(image, ((0, 0), (image.shape[1], image.shape[1])))
        template_samples = np.array([template[pnt[0] - 1, pnt[1] - 1] for pnt in samples_loc])
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

        tic = time.time()
        print("2\tStart evaluation for every affine")

        affines[:, 0, 2] += image.shape[0] / 2 + 1
        affines[:, 1, 2] += image.shape[1] / 2 + 1 + 2 * image.shape[1]
        transformed_samples_loc = np.matmul(affines, samples_loc)  # shape = (#of_configs, 2, #of_samples)
        transformed_samples_loc = transformed_samples_loc.astype(int) - 1  # astype rounding down
        print("transformed_samples_loc min max", np.min(transformed_samples_loc), np.max(transformed_samples_loc))
        image_samples = np.empty((transformed_samples_loc.shape[0], amount_of_points))
        image_samples = image_samples

        print("2\tStart extracting transformed samples from image")
        tic1 = 0
        for i in range(transformed_samples_loc.shape[0]):
            # if i % 100 == 0:
                # tic1 = time.time()
            image_samples[i] = np.array([padded_image[transformed_samples_loc[i, 0, pnt], transformed_samples_loc[i, 0, pnt]]
                                         for pnt in range(transformed_samples_loc.shape[2])])
            # 0 if transformed_samples_loc[i, 0, pnt] >= padded_image.shape[0] or
            #                                               transformed_samples_loc[i, 1, pnt] >= padded_image.shape[1] else
            # if i % 100 == 99:
                # print('1 next 100: {:.8f} seconds'.format(time.time() - tic1), end='')
            # if i % 100 == 0:
                # tic2 = time.time()

            if not self.photometric_invariance:
                score = np.sum(np.abs(template_samples - image_samples[i]))
                # if i % 100 == 99:
                    # print('2 next 100: {:.8f} seconds'.format(time.time() - tic2))
            else:
                sums = (np.sum(template_samples), np.sum(image_samples[i]),
                        np.sum(np.square(template_samples)), np.sum(np.square(image_samples[i])))
                sigma_x = np.sqrt((sums[2] - np.square(sums[0]) / amount_of_points) / amount_of_points) + epsilon
                sigma_y = np.sqrt((sums[3] - np.square(sums[1]) / amount_of_points) / amount_of_points) + epsilon
                sigma_div = sigma_x / sigma_y
                score = np.sum(np.abs(template_samples - (image_samples[i] * sigma_div) + (
                            sums[1] * sigma_div - sums[0]) / amount_of_points))

            distances.append(score / amount_of_points)
        print('2\tDone evaluation for every affine in {:.8f} seconds'.format(time.time() - tic))
        return distances

    def get_good_configs(self, configs, affines, best_distance, new_delta, distances):
        print("3\tin get_good_configs()")
        threshold = best_distance + self.get_threshold(new_delta)
        good_configs = configs[distances <= threshold, :]

        # too many good configs - reducing threshold
        while good_configs.shape[0] > 27000:
            threshold *= 0.99
            good_configs = configs[distances <= threshold, :]

        good_affines = affines[distances <= threshold, :, :]
        if good_configs.shape[0] == 0:
            threshold = np.min(distances)
            good_configs = configs[distances <= threshold, :]
            good_affines = affines[distances <= threshold, :, :]
            if good_configs.shape[0] > 10000:  # all with the same error exactly - probably equivalent
                good_configs = good_configs[:100, :]
                good_affines = good_affines[:100, :, :]
        return good_configs, good_affines

    @staticmethod
    def get_threshold(delta):
        p0 = 0.1341
        p1 = 0.0278
        safety = 0.02
        return p0 * delta + p1 - safety

    def random_expand_configs(self, good_configs, net, level, number_of_points, delta_fact):
        fact = np.power(delta_fact, level)
        new_net = net / fact
        random_vec = np.floor(3 * np.random.rand(number_of_points * good_configs.shape[0], 6) - 1)
        expanded = np.repeat(good_configs, number_of_points, axis=0)
        ranges = np.array([new_net.tx_steps, new_net.ty_steps, new_net.rot_bounds, new_net.sc_bounds, new_net.sc_bounds, new_net.rot_bounds])
        expanded += random_vec * ranges

        return expanded

    @staticmethod
    def get_corners(template_shape, best_affine):
        corners = np.zeros((3, 4))
        corners[0, 2] = corners[0, 3] = template_shape[0]
        corners[1, 1] = corners[1, 2] = template_shape[1]
        corners[2, :] = 1

        transformed_corners = np.matmul(corners, best_affine)
        return [(transformed_corners[0, 0], transformed_corners[1, 0]),
                (transformed_corners[0, 1], transformed_corners[1, 1]),
                (transformed_corners[0, 2], transformed_corners[1, 2]),
                (transformed_corners[0, 3], transformed_corners[1, 3])]
