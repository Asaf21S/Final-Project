from FastMatch import FastMatch
from CreateSamples import random_template
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn
import time
from os import listdir
from csv import writer


def fast_match():
    directory = r"TestImages"
    fm = FastMatch()
    iterations = 2  # later 100 (samples for each new image)

    tic1 = time.time()
    for image_name in listdir(directory):
        if image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg"):
            print("image name:", image_name)
            image = cv2.imread(directory + "\\" + image_name)

            # result_image = image.copy()
            # template = cv2.imread(r"Images\template.png")
            # real_corners = np.array([[147, 212], [214, 188], [267, 35], [200, 60]])

            ml_input = np.empty((0, 39))
            histogram_data = np.empty((0, 103))
            tic2 = time.time()
            for i in range(iterations):
                template, real_corners = random_template(image)
                corners, ml_model_input, histogram_data_samples = fm.run(image, template, real_corners)
                print("Actual corners:")
                print(real_corners[0], real_corners[1], real_corners[2], real_corners[3])
                for lev in range(ml_model_input.shape[0]):
                    ml_model_input[lev, 0] = image_name + "_" + str(i) + "_" + str(ml_model_input[lev, 0])
                    histogram_data_samples[lev, 0] = image_name + "_" + str(i) + "_" + str(histogram_data_samples[lev, 0])
                ml_input = np.vstack([ml_input, ml_model_input])
                histogram_data = np.vstack([histogram_data, histogram_data_samples])

                # cv2.polylines(result_image, [real_corners], True, (0, 255, 0), 1)
                # cv2.polylines(result_image, [corners], True, (255, 0, 0), 1)
            print("\n\n$$$ Total time for the image {}: {:.8f} seconds $$$".format(image_name, time.time() - tic2))

            with open(r'ML_model/Training_Data.csv', 'a') as training_data_file:
                writer(training_data_file).writerows(ml_input)
            with open(r'ML_model/Histogram_Data.csv', 'a') as hist_data_file:
                writer(hist_data_file).writerows(histogram_data)

            ''' Graphs: feature effect on ideal_th
            # check how each feature affect the output
            c = np.linspace(0, 1, ml_input.shape[0])
            plt.figure()
            plt.scatter(ml_input[:, 1], ml_input[:, 0], c=c)    
            plt.figure()
            plt.scatter(ml_input[:, 2], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 3], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 4], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 5], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 6], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 7], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 8], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 9], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 10], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 11], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 12], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 13], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 14], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 15], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 16], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 17], ml_input[:, 0], c=c)
            plt.figure()
            plt.scatter(ml_input[:, 18], ml_input[:, 0], c=c)
            '''

            '''
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
            '''
    print("\n\n-------------------@@@@@ Total time for the images in directory {}: {:.8f} seconds @@@@@-------------------".format(directory, time.time() - tic1))

    ''' import the data from the csv file
    sample_name = np.loadtxt(r'ML_model/Training_Data.csv', dtype='str', delimiter=',', usecols=0)
    sample_y = np.loadtxt(r'ML_model/Training_Data.csv', dtype='float', delimiter=',', usecols=1)
    features = np.loadtxt(r'ML_model/Training_Data.csv', dtype='float', delimiter=',', usecols=tuple(range(2, 39)))
    '''


if __name__ == '__main__':
    fast_match()
