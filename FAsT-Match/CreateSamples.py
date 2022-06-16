import numpy as np
import cv2


def random_template(image, min_scale=0.5, max_scale=2):
    min_dim = np.min(image.shape[:2])  # minimal dimension
    size_fact = 2.5  # maximal relative size template dimension w.r.t. image dimension
    n1 = np.ceil(min_dim / size_fact)
    if n1 % 2 == 0:
        n1 -= 1

    min_tx = 0.5 * ((n1 - 1) * min_scale - (image.shape[1] - 1))
    min_ty = 0.5 * ((n1 - 1) * min_scale - (image.shape[0] - 1))

    range_tx = -2 * min_tx
    range_ty = -2 * min_ty
    range_rot = 2 * np.pi
    range_s = max_scale - min_scale

    ranges = np.array([range_tx, range_ty, range_rot / 4, range_s, range_s, range_rot])
    min_affine = np.array([min_tx, min_ty, -np.pi, min_scale, min_scale, -np.pi])

    attempt = 0
    while True:
        attempt += 1
        rand_weights = np.random.rand(6)
        rand_aff_conf = min_affine + rand_weights * ranges

        # if rotation range isn't [-pi,pi], check rotation range here

        rand_aff_mat = np.empty((2, 3))

        c1c2 = np.cos(rand_aff_conf[5]) * np.cos(rand_aff_conf[2])
        s1s2 = np.sin(rand_aff_conf[5]) * np.sin(rand_aff_conf[2])
        c1s2 = np.cos(rand_aff_conf[5]) * np.sin(rand_aff_conf[2])
        s1c2 = np.sin(rand_aff_conf[5]) * np.cos(rand_aff_conf[2])

        rand_aff_mat[0, 0] = rand_aff_conf[3] * c1c2 - rand_aff_conf[4] * s1s2
        rand_aff_mat[0, 1] = -rand_aff_conf[3] * c1s2 - rand_aff_conf[4] * s1c2
        rand_aff_mat[0, 2] = rand_aff_conf[0]
        rand_aff_mat[1, 0] = rand_aff_conf[3] * s1c2 + rand_aff_conf[4] * c1s2
        rand_aff_mat[1, 1] = rand_aff_conf[4] * c1c2 - rand_aff_conf[3] * s1s2
        rand_aff_mat[1, 2] = rand_aff_conf[1]

        corners = np.ones((3, 4))
        corners[0, 1] = corners[0, 2] = corners[1, 2] = corners[1, 3] = n1
        corners[0] -= n1 / 2 + 0.5
        corners[1] -= n1 / 2 + 0.5

        affine_corners = np.matmul(rand_aff_mat, corners)
        affine_corners[0] += image.shape[1] / 2 + 0.5
        affine_corners[1] += image.shape[0] / 2 + 0.5

        if all(0 < x < image.shape[1] for x in affine_corners[0]) and \
                all(0 < y < image.shape[0] for y in affine_corners[1]):
            # construct the template
            centered_tl3 = np.array([[1, 1], [n1, 1], [n1, n1]])
            src_pts = np.float32(affine_corners.T[:-1, :])
            dst_pts = np.float32(centered_tl3)
            affine = cv2.getAffineTransform(src_pts, dst_pts)
            template = cv2.warpAffine(image, affine, (int(n1), int(n1)))

            # verify that it isn't just flat
            if np.std(template) > 0.1:
                break

        # if attempt > 20:
            # raise Exception("Too many attempts to find random affine template")

    return template, affine_corners.T.astype(int)
