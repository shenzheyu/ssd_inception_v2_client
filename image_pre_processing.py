from timeit import default_timer as timer
import numpy as np
import cv2


def decode_image_opencv(img_path, max_height=800, swapRB=True, imagenet_mean=(0, 0, 0), log=False):
    start = timer()
    image = cv2.imread(img_path, 1)
    if log:
        print("original image shape=", image.shape)
    (h, w) = image.shape[:2]
    if log:
        print("Scale factor=", h / max_height)
    image = image_resize(image, height=max_height)
    org = image
    image = cv2.dnn.blobFromImage(image, scalefactor=1.0, mean=imagenet_mean, swapRB=swapRB, )
    image = np.transpose(image, (0, 2, 3, 1))
    if log:
        print("resized image shape=", image.shape)
    end = timer()
    if log:
        print("decode time=", end - start)
    return image, org


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), h)

    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
