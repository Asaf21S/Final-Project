from MatchNet import MatchNet
import numpy as np
import cv2
from random import randrange
import time
# import matplotlib.pyplot as plt
# import pandas as pd
from sklearn.preprocessing import normalize
from MLP import model_prediction


class FastMatch:
    def __init__(self, epsilon=0.15, delta=0.25, photometric_invariance=False, min_scale=0.5, max_scale=2):
        self.epsilon = epsilon
        self.delta = delta
        self.photometric_invariance = photometric_invariance
        self.min_scale = min_scale
        self.max_scale = max_scale

    def run(self, image, template, real_corners=None, mlp_model=None):
        total_time = time.time()

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
        tic = time.time()
        print("\n----- create_list_of_configs -----")
        configs = self.create_list_of_configs(net)
        print("----- {:.8f} seconds -----".format(time.time() - tic))

        # if configs.shape[0] > 71000000:
            # raise Exception("more than 35 million configs!")

        level = 0
        delta_fact = 1.511
        new_delta = self.delta
        best_dists = np.empty(20)
        num_configs = np.empty(20)
        num_good_configs = np.empty(20)
        orig_percentages = np.empty(20)
        ml_model_input = np.empty((0, 39))
        histogram_data = np.empty((0, 103))

        # preparing input for ML model:
        avg_template = np.mean(template)
        avg_image = np.mean(image)
        avg_ratio = avg_template / avg_image
        x_gradient = cv2.filter2D(template, -1, np.array([[1, -1]], np.float32))[:, 1:]
        y_gradient = cv2.filter2D(template, -1, np.array([[1, -1]], np.float32).T)[1:, :]
        x_gradient = np.mean(np.abs(x_gradient))
        y_gradient = np.mean(np.abs(y_gradient))
        smoothness = (x_gradient + y_gradient) / 2
        template_std = np.std(template)

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
            print("\n----- Level {} configs_to_affines, with {} configs -----".format(level, configs.shape[0]))
            configs, affines = self.configs_to_affines(configs, image.shape, template.shape)
            print("----- {:.8f} seconds -----".format(time.time() - tic))

            # 2] evaluate all configurations
            tic = time.time()
            print("\n----- Level {} evaluate_configs, with {} configs -----".format(level, configs.shape[0]))
            distances = self.evaluate_configs(image, template, affines, samples_loc)
            # modifying affines and samples_loc

            # find the minimum distance
            best_distance = min(distances)
            best_dists[level - 1] = best_distance
            min_index = distances.index(best_distance)
            # best_config = configs[min_index]
            best_affine = affines[min_index]
            print("2\tbestDist = {:.3}".format(best_distance))

            if real_corners is not None:
                corners = self.get_corners(template.shape, image.shape[1], affines)
                dist = corners - real_corners
                dist = np.square(dist)
                dist = np.sum(dist, axis=2)
                dist = np.sqrt(dist)
                dist = np.sum(dist, axis=1)
                correct_index = np.argmin(dist)

                distances_arr = np.array(distances)
                correct_distance = distances[correct_index]
                required_safety = correct_distance - best_distance
                # the safety_window function return a number that should be at least required_safety

                print("min_index", min_index, "correct_index", correct_index)
                print("required_safety", required_safety)

                # preparing input for ML model:
                ideal_th = best_distance + required_safety * 2.0  # this is y/output/label
                distances_mean = np.mean(distances_arr)
                distances_std = np.std(distances_arr)
                max_distance = max(distances)
                distances_range = max_distance - best_distance

                ml_model_input_row = np.array([str(level), ideal_th, image.shape[0], image.shape[1], template.shape[0],
                                               avg_ratio, smoothness, template_std, new_delta, best_distance,
                                               distances_mean, distances_std, np.percentile(distances_arr, 5),
                                               np.percentile(distances_arr, 10), np.percentile(distances_arr, 15),
                                               np.percentile(distances_arr, 20), np.percentile(distances_arr, 25),
                                               np.percentile(distances_arr, 30), np.percentile(distances_arr, 35),
                                               np.percentile(distances_arr, 40), np.percentile(distances_arr, 45),
                                               np.percentile(distances_arr, 50), np.percentile(distances_arr, 55),
                                               np.percentile(distances_arr, 60), np.percentile(distances_arr, 65),
                                               np.percentile(distances_arr, 70), np.percentile(distances_arr, 75),
                                               np.percentile(distances_arr, 80), np.percentile(distances_arr, 85),
                                               np.percentile(distances_arr, 90), np.percentile(distances_arr, 95),
                                               distances_range, len(distances_arr),
                                               best_affine[0, 0], best_affine[0, 1], best_affine[0, 2],
                                               best_affine[1, 0], best_affine[1, 1], best_affine[1, 2]])
                # print(ml_model_input, ml_model_input_row)
                ml_model_input = np.vstack([ml_model_input, ml_model_input_row])

                distances_hist, _ = np.histogram(distances_arr, bins=100)
                histogram_data_row = np.concatenate(([str(level), np.min(distances_arr), np.max(distances_arr)],
                                                     distances_hist))
                histogram_data = np.vstack([histogram_data, histogram_data_row])

                ''' Histograms:
                plt.figure()
                n, bins, patches = plt.hist(distances_arr, bins=100)

                hist_min_distance = float("inf")
                index_of_bar_to_label = 0
                for i, rectangle in enumerate(patches):
                    tmp = abs((rectangle.get_x() + (rectangle.get_width() * (1 / 2))) - correct_distance)
                    if tmp < hist_min_distance:
                        hist_min_distance = tmp
                        index_of_bar_to_label = i
                patches[index_of_bar_to_label].set_color('g')

                print("describe")
                max_percent = 27000 / distances_arr.shape[0]
                if max_percent > 1:
                    max_percent = 1
                percentiles = np.linspace(0, max_percent, 10)
                desc = pd.DataFrame(distances_arr).describe(percentiles=percentiles)
                print(desc, type(desc), desc.shape)
                '''

            mlp_prediction = None
            if mlp_model is not None:
                distances_arr = np.array(distances)
                distances_mean = np.mean(distances_arr)
                distances_std = np.std(distances_arr)
                max_distance = max(distances)
                distances_range = max_distance - best_distance
                mlp_features = np.array([[template.shape[0], avg_ratio, smoothness, template_std, new_delta,
                                         best_distance, distances_mean, distances_std, distances_range,
                                         len(distances_arr)]])
                mlp_features[:, (0, 9)] = normalize(mlp_features[:, (0, 9)])
                mlp_features = np.squeeze(mlp_features)
                mlp_prediction = model_prediction(mlp_model, mlp_features)

            print("----- {:.8f} seconds -----".format(time.time() - tic))

            # 3] choose the 'surviving' configs and delta for next round
            tic = time.time()
            print("\n----- Level {} get_good_configs, with {} configs -----".format(level, configs.shape[0]))
            good_configs, good_affines, orig_percentage = self.get_good_configs(configs, affines, best_distance,
                                                                                new_delta, mlp_prediction, distances)
            print("3\tpercentage of good configs = {:.3}".format(orig_percentage))

            # collect round stats
            num_configs[level - 1] = configs.shape[0]
            num_good_configs[level - 1] = good_configs.shape[0]
            orig_percentages[level - 1] = orig_percentage

            print("----- {:.8f} seconds -----".format(time.time() - tic))

            # 4] break conditions of Branch-and-Bound
            conditions = np.zeros(6, dtype=bool)

            # good enough #1:
            conditions[0] = best_distance < 0.005
            # good enough #2:
            conditions[1] = level > 5 and best_distance < 0.01
            # enough levels:
            conditions[2] = level >= 20
            # no improvement in last 3 rounds:
            conditions[3] = level > 3 and best_distance > ((best_dists[level - 4] + best_dists[level - 3] +
                                                            best_dists[level - 2]) / 3) * 0.97
            # too high expansion rate:
            conditions[4] = level > 2 and good_configs.shape[0] > 1000 and orig_percentage > 0.2
            # a deterioration in the focus:
            conditions[5] = level > 3 and good_configs.shape[0] > 1000 and good_configs.shape[0] > 50 * np.min(
                num_good_configs[:level - 1])

            if any(conditions):
                print("\nBreaking BnB at level {}. Conditions: {}".format(level, conditions))
                print("Best distances by round:       {}".format(best_dists[:level]))
                print("Configs amount per round:      {}".format(num_configs[:level]))
                print("Good configs amount per round: {}".format(num_good_configs[:level]))
                print("Percentage to expand:          {}".format(orig_percentages[:level]))
                break

            # 5] expand 'surviving' configs for next round
            if level == 1 and ((orig_percentage > 0.05 and best_distance > 0.1 and configs.shape[0] < 7.5e6) or
                               (orig_percentage >= 0.01 and best_distance > 0.15 and configs.shape[0] < 5e6)):
                factor = 0.9
                print("\n##### RESTARTING!!! changing from delta: {:.3f}, to delta: {:.3f}".format(new_delta,
                                                                                                   new_delta * factor))
                new_delta *= factor
                level = 0
                net *= factor
                tic = time.time()
                print("\n----- create_list_of_configs -----")
                configs = self.create_list_of_configs(net)
                print('Done create_list_of_configs in {:.8f} seconds'.format(time.time() - tic))
            else:
                print("\n##### CONTINUING!!! prevDelta = {:.3f},  newDelta = {:.3f}".format(new_delta,
                                                                                            new_delta / delta_fact))
                new_delta /= delta_fact
                # expand the good configs
                number_of_points = 80  # amount of new points for each good configuration
                expanded_configs = self.random_expand_configs(good_configs, net, level, number_of_points, delta_fact)
                configs = np.concatenate((good_configs, expanded_configs))

                # remove duplicates
                configs_tuple = [tuple(row) for row in configs]
                configs = np.unique(configs_tuple, axis=0)

                print("*** level {} completed: |goodConfigs| = {}, |expandedConfigs| = {}\n".format(
                    level, good_configs.shape[0], configs.shape[0]))

            # 6] refresh random points
            samples_loc = [(randrange(1, template.shape[0]), randrange(1, template.shape[1])) for _ in range(
                np.rint(num_points).astype(np.int32))]

        corners = self.get_corners(template.shape, image.shape[1], best_affine)[0]
        print('\n\n~~~ Finished FAsT Match in {:.8f} seconds ~~~'.format(time.time() - total_time))
        print("Result corners:")
        print(corners[0], corners[1], corners[2], corners[3])
        # print("ml_model_input", ml_model_input)
        return corners.reshape((-1, 1, 2)), ml_model_input, histogram_data

    @staticmethod
    def create_list_of_configs(net: MatchNet):
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

        grid_size = ntx_steps * nty_steps * np.square(ns_steps) * nr_steps * nr2_steps

        # iterate through each possible affine configuration steps
        configs = np.empty((grid_size, 6))
        grid_ind = 0
        for tx_ind in range(ntx_steps):
            tx = tx_steps[tx_ind]
            # print("step {} out of {}".format(tx_ind + 1, ntx_steps))
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

        return configs

    @staticmethod
    def configs_to_affines(configs, im_shape, te_shape):
        affines = np.empty((configs.shape[0], 2, 3))

        c1c2 = np.cos(configs[:, 5]) * np.cos(configs[:, 2])
        s1s2 = np.sin(configs[:, 5]) * np.sin(configs[:, 2])
        c1s2 = np.cos(configs[:, 5]) * np.sin(configs[:, 2])
        s1c2 = np.sin(configs[:, 5]) * np.cos(configs[:, 2])

        affines[:, 0, 0] = configs[:, 3] * c1c2 - configs[:, 4] * s1s2
        affines[:, 0, 1] = -configs[:, 3] * c1s2 - configs[:, 4] * s1c2
        affines[:, 0, 2] = configs[:, 0]
        affines[:, 1, 0] = configs[:, 3] * s1c2 + configs[:, 4] * c1s2
        affines[:, 1, 1] = configs[:, 4] * c1c2 - configs[:, 3] * s1s2
        affines[:, 1, 2] = configs[:, 1]

        # filter configurations which fall outside of image boundaries
        corners = np.ones((3, 4))
        corners[0, 1] = corners[0, 2] = te_shape[0]
        corners[1, 2] = corners[1, 3] = te_shape[1]
        corners[0] -= te_shape[0] / 2 + 0.5
        corners[1] -= te_shape[1] / 2 + 0.5

        affine_corners = np.matmul(affines, corners)
        affine_corners[:, 0] += im_shape[0] / 2 + 0.5
        affine_corners[:, 1] += im_shape[1] / 2 + 0.5

        print("1\tchecking the transformed corners")
        insiders = np.zeros(configs.shape[0])
        for i in range(configs.shape[0]):
            # if all(-10 < x < im_shape[0] + 10 for x in affine_corners[i, 0]) and
            # all(-10 < y < im_shape[1] + 10 for y in affine_corners[i, 1]):
            if all(0 < x < im_shape[0] for x in affine_corners[i, 0]) and \
                    all(0 < y < im_shape[1] for y in affine_corners[i, 1]):
                insiders[i] = 1

        configs = configs[insiders == 1, :]
        affines = affines[insiders == 1, :, :]
        print("1\treduced amount of configs to", configs.shape[0])
        return configs, affines

    def evaluate_configs(self, image, template, affines, samples_loc):
        amount_of_points = len(samples_loc)

        padded_image = np.pad(image, ((0, 0), (image.shape[1], image.shape[1])))
        samples_loc = np.array(samples_loc)
        template_samples = template[samples_loc[:, 0] - 1, samples_loc[:, 1] - 1]
        samples_loc = samples_loc.T
        samples_loc[0] -= int((template.shape[0] + 1) / 2)
        samples_loc[1] -= int((template.shape[1] + 1) / 2)
        samples_loc = np.vstack([samples_loc, [1.0] * amount_of_points])

        epsilon = 1.0e-7
        print("2\tStart evaluation for every affine")

        affines[:, 0, 2] += image.shape[0] / 2 + 1
        affines[:, 1, 2] += image.shape[1] / 2 + 1 + 1 * image.shape[1]

        if not self.photometric_invariance:
            score = np.array([])
            print("2\tStart extracting transformed samples from image")
            partition = 100000
            for i in range(int(np.ceil(affines.shape[0] / partition))):
                if partition * (i + 1) > affines.shape[0]:
                    transformed_samples_loc = np.matmul(affines[partition * i:, :, :], samples_loc)
                else:
                    transformed_samples_loc = np.matmul(affines[partition * i:partition * (i + 1), :, :],
                                                        samples_loc)  # shape = (partition, 2, #of_samples)
                transformed_samples_loc = transformed_samples_loc.astype(int) - 1  # astype rounding down

                image_samples = padded_image[transformed_samples_loc[:, 0, :], transformed_samples_loc[:, 1, :]]
                score = np.append(score, np.sum(np.abs(template_samples - image_samples[:]), axis=1))
        else:
            transformed_samples_loc = np.matmul(affines, samples_loc)  # shape = (#of_configs, 2, #of_samples)
            transformed_samples_loc = transformed_samples_loc.astype(int) - 1  # astype rounding down

            print("2\tStart extracting transformed samples from image")
            image_samples = padded_image[transformed_samples_loc[:, 0, :], transformed_samples_loc[:, 1, :]]

            sums = (np.sum(template_samples), np.sum(image_samples[:], axis=1),
                    np.sum(np.square(template_samples)), np.sum(np.square(image_samples[:]), axis=1))
            sigma_x = np.sqrt((sums[2] - np.square(sums[0]) / amount_of_points) / amount_of_points) + epsilon
            sigma_y = np.sqrt((sums[3] - np.square(sums[1]) / amount_of_points) / amount_of_points) + epsilon
            sigma_div = sigma_x / sigma_y
            score = np.sum(np.abs(template_samples - (image_samples[:] * sigma_div) + (
                    sums[1] * sigma_div - sums[0]) / amount_of_points), axis=1)

        distances = list(score / amount_of_points)
        return distances

    def get_good_configs(self, configs, affines, best_distance, new_delta, mlp_prediction, distances):
        if mlp_prediction is None:
            threshold = best_distance + self.safety_window(new_delta)
        else:
            threshold = mlp_prediction

        good_configs = configs[distances <= threshold, :]
        orig_percentage = good_configs.shape[0] / configs.shape[0]

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
        return good_configs, good_affines, orig_percentage

    @staticmethod
    def safety_window(delta):
        p0 = 0.1341
        p1 = 0.0278
        safety = 0.02
        return p0 * delta + p1 - safety

    @staticmethod
    def random_expand_configs(good_configs, net, level, number_of_points, delta_fact):
        fact = np.power(delta_fact, level)
        new_net = net / fact
        random_vec = np.floor(3 * np.random.rand(number_of_points * good_configs.shape[0], 6) - 1)
        expanded = np.repeat(good_configs, number_of_points, axis=0)
        ranges = np.array([new_net.tx_steps, new_net.ty_steps, new_net.rot_steps,
                           new_net.sc_steps, new_net.sc_steps, new_net.rot_steps])
        expanded += random_vec * ranges

        return expanded

    @staticmethod
    def get_corners(template_shape, image_shape1, affine_mat):
        corners = np.ones((3, 4))
        corners[0, 1] = corners[0, 2] = template_shape[0]
        corners[1, 2] = corners[1, 3] = template_shape[1]
        corners[0] -= template_shape[0] / 2 + 0.5
        corners[1] -= template_shape[1] / 2 + 0.5

        transformed_corners = np.matmul(affine_mat, corners)
        if affine_mat.ndim == 2:
            transformed_corners = np.array([transformed_corners])

        transformed_corners[:, 1] -= image_shape1
        transformed_corners[:, [0, 1]] = transformed_corners[:, [1, 0]]
        transformed_corners = np.transpose(transformed_corners, axes=(0, 2, 1)).astype(np.int32)
        return transformed_corners
