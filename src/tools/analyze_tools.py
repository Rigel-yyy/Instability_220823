from .type_tools import ScalarField2D
import numpy as np
import cv2 as cv


class Image:
    def __init__(self, mat: ScalarField2D):
        self.data = mat
        self.binary_image: {None or BinaryImage} = None
        self.grey_image: {None or ScalarField2D} = None

    def get_binary_image(self,
                         low_bound: float,
                         up_bound: float,
                         update: bool = False):
        """
        return image as binary image between two bounds
        -----------
        low_bound: float,
        up_bound: float,
        update: bool = False
            force update on self.binary_image when update == True
        """
        if self.binary_image is None or update:
            sel = (self.data > low_bound) & (self.data < up_bound)
            self.binary_image = BinaryImage(np.where(sel, 1, 0), low_bound, up_bound)
        return self.binary_image

    def get_grey_image(self, update: bool = False):
        """
        map a 2D scalar field to a grey scale image
        """
        if self.grey_image is None or update:
            _range = np.max(self.data) - np.min(self.data)
            self.grey_image = ((self.data - np.min(self.data)) * 255 / _range) \
                              .astype(np.uint8)
        return self.grey_image

    def get_edge(self):
        img = self.get_grey_image()
        return cv.Canny(img, 100, 150)

    def get_front_most(self, direction="right"):
        """
        get front end index of current grey image
        :param direction:
        :return: find edge -> int, edge not found -> None
        """
        edge = self.get_edge()
        if np.sum(edge) >= 255 * 2:
            if direction == "right":
                return np.max(np.argwhere(np.count_nonzero(edge, axis=0) >= 2))
            elif direction == "left":
                return np.min(np.argwhere(np.count_nonzero(edge, axis=0) >= 2))
        else:
            return None


class BinaryImage:
    def __init__(self, mat: ScalarField2D,
                 low_bound: float,
                 up_bound: float):
        self.data = mat
        self.low_bound = low_bound
        self.up_bound = up_bound

    def get_shape_center(self):
        """
        select image between low_bound and up_bound
        calculate the center of the shape
        """
        shape = self.data.shape
        y, x = np.indices(shape)
        x_center = np.sum(x * self.data) / np.sum(self.data)
        y_center = np.sum(y * self.data) / np.sum(self.data)
        return x_center, y_center

    def get_x_width(self):
        return np.sum(self.data, axis=1).mean()

    def get_y_width(self):
        return np.sum(self.data, axis=0).mean()
