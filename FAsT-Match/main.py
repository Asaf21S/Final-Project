from FastMatch import FastMatch
from CreateSamples import random_template
import numpy as np
import matplotlib.pyplot as plt
import cv2
# import sklearn
import time
from os import listdir
from csv import writer


def example_run(image, template, real_corners):
    fm = FastMatch()
    result_image = image.copy()
    corners, _, _ = fm.run(image, template, real_corners)
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
                corners, ml_model_input, histogram_data_samples = fm.run(image, template, real_corners)
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


def show_features(image_directory):
    # import the data from the csv file
    sample_name = np.loadtxt(image_directory, dtype='str', delimiter=',', usecols=0)
    sample_y = np.loadtxt(image_directory, dtype='float', delimiter=',', usecols=1)
    features = np.loadtxt(image_directory, dtype='float', delimiter=',', usecols=tuple(range(2, 39)))
    print(sample_name.shape, sample_y.shape, features.shape)

    # Graphs: feature effect on ideal_th
    # check how each feature affect the output
    c = np.linspace(0, 1, features.shape[0])
    for feature in range(features.shape[1]):
        plt.figure()
        plt.scatter(features[:, feature], sample_y[:], c=c)

    plt.show()


if __name__ == '__main__':
    images_folder = r"Images/Images1"
    features_csv = r"ML_model/Training_Data1.csv"
    histogram_csv = r"ML_model/Histogram_Data1.csv"
    templates_per_image = 5

    # ex_image = cv2.imread(r"TestImages\image.png")
    # ex_template = cv2.imread(r"TestImages\template.png")
    # ex_real_corners = np.array([[147, 212], [214, 188], [267, 35], [200, 60]])
    # example_run(ex_image, ex_template, ex_real_corners)
    fast_match(templates_per_image, images_folder, features_csv, histogram_csv)
    # show_features(images_folder)
