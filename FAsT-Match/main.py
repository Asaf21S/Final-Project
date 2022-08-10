from FastMatch import FastMatch
from CreateSamples import random_template
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import time
from os import listdir
from csv import writer
import MLP


def example_run(image, template, real_corners):
    fm = FastMatch()
    result_image = image.copy()
    corners, _, _ = fm.run(image, template, real_corners=real_corners)
    print("Actual corners:")
    print(real_corners[0], real_corners[1], real_corners[2], real_corners[3])

    cv2.polylines(result_image, [real_corners], True, (0, 255, 0), 1)
    cv2.polylines(result_image, [corners], True, (255, 0, 0), 1)

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title("image")
    plt.figure()
    plt.imshow(template, cmap='gray')
    plt.title("template")
    plt.figure()
    plt.imshow(result_image)
    plt.title("result")

    plt.show()


def fast_match(iterations, image_directory, features_file, histogram_file):
    fm = FastMatch()
    tic1 = time.time()
    for image_name in listdir(image_directory):
        if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
            image = cv2.imread(image_directory + "\\" + image_name)

            ml_input = np.empty((0, 39))
            histogram_data = np.empty((0, 103))
            tic2 = time.time()
            for i in range(iterations):
                print("Image name:", image_name, ". Template number:", str(i + 1))
                template, real_corners = random_template(image)
                corners, ml_model_input, histogram_data_samples = fm.run(image, template, real_corners=real_corners)
                print("Actual corners:")
                print(real_corners[0], real_corners[1], real_corners[2], real_corners[3])
                for lev in range(ml_model_input.shape[0]):
                    ml_model_input[lev, 0] = image_name + "_" + str(i) + "_" + str(ml_model_input[lev, 0])
                    histogram_data_samples[lev, 0] = image_name + "_" + str(i) + "_" + str(histogram_data_samples[lev, 0])
                ml_input = np.vstack([ml_input, ml_model_input])
                histogram_data = np.vstack([histogram_data, histogram_data_samples])
            print("\n\n$$$ Total time for the image {}: {:.8f} seconds $$$".format(image_name, time.time() - tic2))

            with open(features_file, 'a') as training_data_file:
                writer(training_data_file).writerows(ml_input)
            with open(histogram_file, 'a') as hist_data_file:
                writer(hist_data_file).writerows(histogram_data)
    print("\n\n-------------------@@@@@ Total time for the images in directory {}: {:.8f} seconds @@@@@-------------------".format(image_directory, time.time() - tic1))


def show_features(y, x):
    # Graphs: feature effect on ideal_th
    # check how each feature affect the output
    # c = np.linspace(0, 1, features.shape[0])
    for feature in range(x.shape[1]):
        plt.figure()
        plt.scatter(x[:, feature], y[:], alpha=0.1)

    plt.show()


def corner_dist(corners1, corners2):
    corners1_copy = np.copy(corners1)
    corners1_copy = np.squeeze(corners1_copy)
    dist1 = corners1_copy - corners2
    dist1 = np.square(dist1)
    dist1 = np.sum(dist1, axis=1)
    dist1 = np.sqrt(dist1)
    dist1 = np.sum(dist1)

    corners1_copy[[1, 3]] = corners1_copy[[3, 1]]
    dist2 = corners1_copy - corners2
    dist2 = np.square(dist2)
    dist2 = np.sum(dist2, axis=1)
    dist2 = np.sqrt(dist2)
    dist2 = np.sum(dist2)

    return min([dist1, dist2])


def check_model(iterations, image, model):
    fm = FastMatch()
    corners_distance_with = []
    time_with = []
    corners_distance_without = []
    time_without = []
    result_image = image.copy()
    for i in range(iterations):
        template, real_corners = random_template(image)

        tic = time.time()
        corners_without, _, _ = fm.run(image, template)
        print("Actual corners:")
        print(real_corners[0], real_corners[1], real_corners[2], real_corners[3])
        time_without.append(time.time() - tic)
        corners_distance_without.append(corner_dist(corners_without, real_corners))

        tic = time.time()
        corners_with, _, _ = fm.run(image, template, mlp_model=model)
        print("Actual corners:")
        print(real_corners[0], real_corners[1], real_corners[2], real_corners[3])
        time_with.append(time.time() - tic)
        corners_distance_with.append(corner_dist(corners_with, real_corners))

        cv2.polylines(result_image, [real_corners], True, (0, 255, 0), 1)
        cv2.polylines(result_image, [corners_without], True, (255, 0, 0), 1)
        cv2.polylines(result_image, [corners_with], True, (138, 43, 226), 1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("image")
    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title("result")

    plt.figure()
    plt.scatter(range(1, iterations + 1), time_without, c="red")
    plt.scatter(range(1, iterations + 1), time_with, c="magenta")
    plt.title("time")
    plt.figure()
    plt.scatter(range(1, iterations + 1), corners_distance_without, c="red")
    plt.scatter(range(1, iterations + 1), corners_distance_with, c="magenta")
    plt.title("accuracy")
    print(corners_distance_with, time_with, corners_distance_without, time_without)
    plt.show()


def import_model_results(results_path):
    all_corners_distance_1 = np.zeros((7, 0))
    all_times_1 = np.zeros((7, 0))
    all_corners_distance_2 = np.zeros((8, 0))
    all_times_2 = np.zeros((8, 0))
    all_corners_distance_3 = np.zeros((11, 0))
    all_times_3 = np.zeros((11, 0))
    all_corners_distance_4 = np.zeros((16, 0))
    all_times_4 = np.zeros((16, 0))

    for file_name in listdir(results_path):
        if file_name.endswith(".npy"):
            if file_name.startswith("corners_distance1"):
                corners_distance = np.load(results_path + "/" + file_name)
                all_corners_distance_1 = np.concatenate((all_corners_distance_1, corners_distance), axis=1)
            elif file_name.startswith("corners_distance2"):
                corners_distance = np.load(results_path + "/" + file_name)
                all_corners_distance_2 = np.concatenate((all_corners_distance_2, corners_distance), axis=1)
            elif file_name.startswith("corners_distance3"):
                corners_distance = np.load(results_path + "/" + file_name)
                all_corners_distance_3 = np.concatenate((all_corners_distance_3, corners_distance), axis=1)
            elif file_name.startswith("corners_distance4"):
                corners_distance = np.load(results_path + "/" + file_name)
                all_corners_distance_4 = np.concatenate((all_corners_distance_4, corners_distance), axis=1)
            elif file_name.startswith("times1"):
                times = np.load(results_path + "/" + file_name)
                all_times_1 = np.concatenate((all_times_1, times), axis=1)
            elif file_name.startswith("times2"):
                times = np.load(results_path + "/" + file_name)
                all_times_2 = np.concatenate((all_times_2, times), axis=1)
            elif file_name.startswith("times3"):
                times = np.load(results_path + "/" + file_name)
                all_times_3 = np.concatenate((all_times_3, times), axis=1)
            elif file_name.startswith("times4"):
                times = np.load(results_path + "/" + file_name)
                all_times_4 = np.concatenate((all_times_4, times), axis=1)
    return (all_corners_distance_1, all_corners_distance_2, all_corners_distance_3, all_corners_distance_4),\
           (all_times_1, all_times_2, all_times_3, all_times_4)


def view_model_results(corners_distance, times):
    avg_acc_1 = np.average(corners_distance[0], axis=1)
    avg_tme_1 = np.average(times[0], axis=1)
    med_acc_1 = np.median(corners_distance[0], axis=1)
    med_tme_1 = np.median(times[0], axis=1)

    avg_acc_2 = np.average(corners_distance[1], axis=1)
    avg_tme_2 = np.average(times[1], axis=1)
    med_acc_2 = np.median(corners_distance[1], axis=1)
    med_tme_2 = np.median(times[1], axis=1)

    avg_acc_3 = np.average(corners_distance[2], axis=1)
    avg_tme_3 = np.average(times[2], axis=1)
    med_acc_3 = np.median(corners_distance[2], axis=1)
    med_tme_3 = np.median(times[2], axis=1)

    avg_acc_4 = np.average(corners_distance[3], axis=1)
    avg_tme_4 = np.average(times[3], axis=1)
    med_acc_4 = np.median(corners_distance[3], axis=1)
    med_tme_4 = np.median(times[3], axis=1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(np.array(["None", "0.5", "0.8", "1.0", "1.2", "1.5", "2.0"]), avg_acc_1, color='orange', label="Average")
    plt.bar(np.array(["None", "0.5", "0.8", "1.0", "1.2", "1.5", "2.0"]), med_acc_1, width=0.6, color='red',
            label="Median")
    plt.xticks(rotation=-45)
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.bar(np.array(["None", "0.5", "0.8", "1.0", "1.2", "1.5", "2.0"]), avg_tme_1, color='orange', label="Average")
    plt.bar(np.array(["None", "0.5", "0.8", "1.0", "1.2", "1.5", "2.0"]), med_tme_1, width=0.6, color='red',
            label="Median")
    plt.xticks(rotation=-45)
    plt.legend()
    plt.title("Time")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(np.array(["None", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]), avg_acc_2, color='orange',
            label="Average")
    plt.bar(np.array(["None", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]), med_acc_2, width=0.6, color='red',
            label="Median")
    plt.xticks(rotation=-45)
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.bar(np.array(["None", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]), avg_tme_2, color='orange',
            label="Average")
    plt.bar(np.array(["None", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]), med_tme_2, width=0.6, color='red',
            label="Median")
    plt.xticks(rotation=-45)
    plt.legend()
    plt.title("Time")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(np.array(["None", "0.36", "0.38", "0.4", "0.42", "0.44", "0.46", "0.48", "0.5", "0.52", "0.54"]), avg_acc_3,
            color='orange', label="Average")
    plt.bar(np.array(["None", "0.36", "0.38", "0.4", "0.42", "0.44", "0.46", "0.48", "0.5", "0.52", "0.54"]), med_acc_3,
            width=0.6, color='red', label="Median")
    plt.xticks(rotation=-45)
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.bar(np.array(["None", "0.36", "0.38", "0.4", "0.42", "0.44", "0.46", "0.48", "0.5", "0.52", "0.54"]), avg_tme_3,
            color='orange', label="Average")
    plt.bar(np.array(["None", "0.36", "0.38", "0.4", "0.42", "0.44", "0.46", "0.48", "0.5", "0.52", "0.54"]), med_tme_3,
            width=0.6, color='red', label="Median")
    plt.xticks(rotation=-45)
    plt.legend()
    plt.title("Time")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(np.array(["None", "0.15", "0.165", "0.1815", "0.19965", "0.219615", "0.241577", "0.265734", "0.292308",
                      "0.321538", "0.353692", "0.389061", "0.427968", "0.470764", "0.517841", "0.569625"]), avg_acc_4,
            color='orange', label="Average")
    plt.bar(np.array(["None", "0.15", "0.165", "0.1815", "0.19965", "0.219615", "0.241577", "0.265734", "0.292308",
                      "0.321538", "0.353692", "0.389061", "0.427968", "0.470764", "0.517841", "0.569625"]), med_acc_4,
            width=0.6, color='red', label="Median")
    plt.xticks(rotation=-45)
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.bar(np.array(["None", "0.15", "0.165", "0.1815", "0.19965", "0.219615", "0.241577", "0.265734", "0.292308",
                      "0.321538", "0.353692", "0.389061", "0.427968", "0.470764", "0.517841", "0.569625"]), avg_tme_4,
            color='orange', label="Average")
    plt.bar(np.array(["None", "0.15", "0.165", "0.1815", "0.19965", "0.219615", "0.241577", "0.265734", "0.292308",
                      "0.321538", "0.353692", "0.389061", "0.427968", "0.470764", "0.517841", "0.569625"]), med_tme_4,
            width=0.6, color='red', label="Median")
    plt.xticks(rotation=-45)
    plt.legend()
    plt.title("Time")

    colors = cm.rainbow(np.linspace(0, 1, corners_distance[0].shape[0] - 1))
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(range(corners_distance[0].shape[1]), corners_distance[0][0], s=10, color='k', label="None")
    for y, c, mod in zip(corners_distance[0][1:], colors, np.array(["0.5", "0.8", "1.0", "1.2", "1.5", "2.0"])):
        plt.scatter(range(corners_distance[0].shape[1]), y, s=10, color=c, label="model " + mod)
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(range(corners_distance[0].shape[1]), corners_distance[0][0], s=10, color='k', label="None")
    for y, c, mod in zip(corners_distance[0][1:], colors, np.array(["0.5", "0.8", "1.0", "1.2", "1.5", "2.0"])):
        plt.scatter(range(corners_distance[0].shape[1]), y, s=10, color=c, label="model " + mod)
    ax = plt.gca()
    ax.set_ylim([0, 60])
    plt.title("Limited Accuracy")
    plt.legend()

    colors = cm.rainbow(np.linspace(0, 1, corners_distance[1].shape[0] - 1))
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(range(corners_distance[1].shape[1]), corners_distance[1][0], s=10, color='k', label="None")
    for y, c, mod in zip(corners_distance[1][1:], colors, np.array(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])):
        plt.scatter(range(corners_distance[1].shape[1]), y, s=10, color=c, label="model " + mod)
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(range(corners_distance[1].shape[1]), corners_distance[1][0], s=10, color='k', label="None")
    for y, c, mod in zip(corners_distance[1][1:], colors, np.array(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])):
        plt.scatter(range(corners_distance[1].shape[1]), y, s=10, color=c, label="model " + mod)
    ax = plt.gca()
    ax.set_ylim([0, 60])
    plt.title("Limited Accuracy")
    plt.legend()

    colors = cm.rainbow(np.linspace(0, 1, corners_distance[2].shape[0] - 1))
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(range(corners_distance[2].shape[1]), corners_distance[2][0], s=10, color='k', label="None")
    for y, c, mod in zip(corners_distance[2][1:], colors,
                         np.array(["0.36", "0.38", "0.4", "0.42", "0.44", "0.46", "0.48", "0.5", "0.52", "0.54"])):
        plt.scatter(range(corners_distance[2].shape[1]), y, s=10, color=c, label="model " + mod)
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(range(corners_distance[2].shape[1]), corners_distance[2][0], s=10, color='k', label="None")
    for y, c, mod in zip(corners_distance[2][1:], colors,
                         np.array(["0.36", "0.38", "0.4", "0.42", "0.44", "0.46", "0.48", "0.5", "0.52", "0.54"])):
        plt.scatter(range(corners_distance[2].shape[1]), y, s=10, color=c, label="model " + mod)
    ax = plt.gca()
    ax.set_ylim([0, 60])
    plt.title("Limited Accuracy")
    plt.legend()

    colors = cm.rainbow(np.linspace(0, 1, corners_distance[3].shape[0] - 1))
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(range(corners_distance[3].shape[1]), corners_distance[3][0], s=10, color='k', label="None")
    for y, c, mod in zip(corners_distance[3][1:], colors,
                         np.array(["0.15", "0.165", "0.1815", "0.19965", "0.219615",
                                   "0.241577", "0.265734", "0.292308", "0.321538", "0.353692",
                                   "0.389061", "0.427968", "0.470764", "0.517841", "0.569625"])):
        if mod.startswith("0.26") or mod.startswith("0.42"):
            plt.scatter(range(corners_distance[3].shape[1]), y, s=10, color=c, label="model " + mod)
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(range(corners_distance[3].shape[1]), corners_distance[3][0], s=10, color='k', label="None")
    for y, c, mod in zip(corners_distance[3][1:], colors,
                         np.array(["0.15", "0.165", "0.1815", "0.19965", "0.219615",
                                   "0.241577", "0.265734", "0.292308", "0.321538", "0.353692",
                                   "0.389061", "0.427968", "0.470764", "0.517841", "0.569625"])):
        if mod.startswith("0.26") or mod.startswith("0.42"):
            plt.scatter(range(corners_distance[3].shape[1]), y, s=10, color=c, label="model " + mod)
    ax = plt.gca()
    ax.set_ylim([0, 60])
    plt.title("Limited Accuracy")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    images_folder = r"Images/Images1"
    model_data_folder = r"ML_model"
    features_csv = r"ML_model/Training_Data1.csv"
    histogram_csv = r"ML_model/Histogram_Data1.csv"
    models_path = r"PyTorch_models"
    templates_per_image = 5

    # ex_image = cv2.imread(r"TestImages\image.png")
    # ex_template = cv2.imread(r"TestImages\template.png")
    # ex_real_corners = np.array([[147, 212], [214, 188], [267, 35], [200, 60]])
    # example_run(ex_image, ex_template, ex_real_corners)

    # fast_match(templates_per_image, images_folder, features_csv, histogram_csv)

    # names_data, ideal_th, features = MLP.import_data(model_data_folder)
    # names_hist, edges, bins = MLP.import_histogram(model_data_folder)
    # factor = 1.0 / 2
    # processed_features, processed_ideal_th, processed_bins = MLP.preprocess(features, ideal_th, factor, bins)

    # show_features(processed_ideal_th, processed_features)

    # mlp_model = MLP.load_model(models_path + "/mlp0.5.pth")
    # img = cv2.imread(r"Images\Images2\zurich_object0024.view05.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # check_model(3, img, mlp_model)

    accuracy_results, time_results = import_model_results(models_path)
    view_model_results(accuracy_results, time_results)
