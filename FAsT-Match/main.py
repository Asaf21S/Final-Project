from FastMatch import FastMatch
from CreateSamples import random_template
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn


def fast_match():
    image = cv2.imread(r"Images\symm_arch_2.jpg")
    result_image = image.copy()

    # template = cv2.imread(r"Images\template.png")
    # real_corners = np.array([[147, 212], [214, 188], [267, 35], [200, 60]])

    fm = FastMatch()
    ml_input = np.empty((0, 8))
    iterations = 3  # later 100 (samples for each new image)
    for i in range(iterations):
        template, real_corners = random_template(image)
        corners, ml_model_input = fm.run(image, template, real_corners)
        print("Actual corners:")
        print(real_corners[0], real_corners[1], real_corners[2], real_corners[3])
        ml_input = np.vstack([ml_input, ml_model_input])

        cv2.polylines(result_image, [real_corners], True, (0, 255, 0), 1)
        cv2.polylines(result_image, [corners], True, (255, 0, 0), 1)

    with open(r'ML_model/Training_Data.csv', 'a') as training_data:
        np.savetxt(training_data, ml_input, delimiter=',',
                   fmt=['%.12f', '%.12f', '%.12f', '%i', '%i', '%i', '%i', '%i'], comments='')

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


if __name__ == '__main__':
    fast_match()
