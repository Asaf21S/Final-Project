from FastMatch import FastMatch
from CreateSamples import random_template
import numpy as np
import matplotlib.pyplot as plt
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
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("image")
    plt.subplot(2, 2, 2)
    plt.imshow(template, cmap='gray')
    plt.title("template")
    plt.subplot(2, 2, 3)
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
    for i in range(iterations):
        result_image = image.copy()
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
        plt.subplot(2, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("image")
        plt.subplot(2, 2, 2)
        plt.imshow(template, cmap='gray')
        plt.title("template")
        plt.subplot(2, 2, 3)
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
    # factor = 0.5 / 2
    # processed_features, processed_ideal_th, processed_bins = MLP.preprocess(features, ideal_th, factor, bins)

    mlp_model = MLP.load_model(models_path + "/mlp0.5.pth")
    img = cv2.imread(r"Images\Images2\turtle.jpg")
    check_model(5, img, mlp_model)
