from FastMatch import FastMatch
import matplotlib.pyplot as plt
import cv2


def fast_match():
    image = cv2.imread(r'Images\image1.tif')
    template = cv2.imread(r'Images\image1.tif')

    fm = FastMatch()
    corners = fm.run(image, template)
    result_image = image.copy()
    cv2.polylines(result_image, corners, True, (255, 255, 255), 1)

    '''
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("image")
    plt.subplot(2, 2, 2)
    plt.imshow(template, cmap='gray')
    plt.title("template")
    plt.subplot(2, 2, 3)
    plt.imshow(result_image, cmap='gray')
    plt.title("result")

    plt.show()
    '''


if __name__ == '__main__':
    fast_match()
