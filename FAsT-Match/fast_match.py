from scipy.io import loadmat
from numpy import ones, pi


class SearchRange:
    def __init__(self):
        self.min_scale = 0.5
        self.max_scale = 2
        self.min_rotation = -pi
        self.max_rotation = pi
        self.min_tx = None
        self.max_tx = None
        self.min_ty = None
        self.max_ty = None


def make_odd(im):
    """
    make all 3 dimensions odd
    :param im: 3 dimension matrix
    :return:
        cropped_im
        cropped_h: how mach was cropped in height
        cropped_w: how mach was cropped in width
    """
    shape = im.shape
    if len(shape) < 3:
        shape = list(shape) + [1]
    h, w, d = shape
    cropped_h = 0
    cropped_w = 0
    if h % 2 == 0:
        cropped_h = 1
        im = im[:-1]
    if w % 2 == 0:
        cropped_w = 1
        im = im[:, :-1]
    if d % 2 == 0:
        im = im[:, :, :-1]
    return im, cropped_h, cropped_w


def fast_match(template,
               img,
               template_mask=None,
               epsilon=0.15,
               delta=0.25,
               photometric_invariance=0,
               search_range=SearchRange()):
    if template_mask is None:
        template_mask = ones(template.shape)
    img, _, _ = make_odd(img)
    template, _, _ = make_odd(template)
    template_mask, _, _ = make_odd(template_mask)
    img = img.astype("double")
    template = template.astype("double")

    if min(img.min(), template.min()) < -0.1 or max(img.max(), template.max()) > 1.1:
        raise Exception("FastMatch: img and template should both be of class 'double' (in the approx range of [0,1])")

    if img.shape[2] != template.shape[2]:
        raise Exception('img and template should both be of same dimension per pixel')


def test_alg():
    template = loadmat('template.mat')['template']
    img = loadmat('img.mat')['img']
    fast_match(template, img)


if __name__=="__main__":
    test_alg()