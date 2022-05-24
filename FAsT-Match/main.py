from FastMatch import FastMatch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sklearn


def fast_match():
    image = cv2.imread(r"Images\image.png")
    template = cv2.imread(r"Images\template.png")

    fm = FastMatch()
    real_corners = np.array([[147, 212], [214, 188], [267, 35], [200, 60]])
    corners = fm.run(image, template, real_corners)
    result_image = image.copy()
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


if __name__ == '__main__':
    fast_match()
