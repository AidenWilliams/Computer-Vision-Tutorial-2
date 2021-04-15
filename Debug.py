import numpy as np

import cv2


class Window(object):
    def __init__(self, image, n, s):
        # make s can be tuple
        self.x_boundary = image.shape[1] + n
        self.y_boundary = image.shape[0] + s
        self.top_left = (0, 0)
        self.bot_right = (n, n)
        self.previousBotY = n
        self.height = n
        self.stride = (s, s)
        try:
            self.channels = image.shape[2]
        except:
            self.channels = 1

    def resetPos(self):
        self.top_left = (0, 0)
        self.bot_right = (self.height, self.height)

    def forwardPos(self):
        # Case when you need to go down and start new line
        if (self.bot_right + self.stride)[0] >= (self.x_boundary - self.height):
            # print("down")
            return (0, self.top_left[1] + self.stride[1]), (self.height, self.bot_right[1] + self.stride[1])
        # generic move right case
        else:
            return (self.top_left[0] + self.stride[0], self.top_left[1]), \
                   (self.bot_right[0] + self.stride[0], self.bot_right[1])

    def forwardMove(self):
        self.top_left, self.bot_right = self.forwardPos()
        return self.top_left, self.bot_right

    def inBoundary(self, new_top_left=None, new_bot_right=None):
        if new_top_left is None:
            new_top_left = self.top_left
        if new_bot_right is None:
            new_bot_right = self.bot_right

        return new_bot_right[0] <= self.x_boundary and new_bot_right[1] <= self.y_boundary and \
               new_top_left[0] >= 0 and new_top_left[1] >= 0

    def changedY(self):
        if self.previousBotY == self.bot_right[1]:
            return False
        else:
            self.previousBotY = self.bot_right[1]
            return True

    # add get image in boundary
    def getImageInBoundary(self, image):
        new_image = []
        for i in range(self.top_left[1], self.bot_right[1]):
            if i >= image.shape[0]:
                continue
            new_image.append(image[i][self.top_left[0]: self.bot_right[0]])

        if self.channels == 1:
            x = np.resize(np.array(new_image), (self.height, self.height))
        else:
            x = np.resize(np.array(new_image), (self.height, self.height, self.channels))
        return x

    def __str__(self):
        return "Top Left Corner " + str(self.top_left) + "\nBot Right Corner " + str(self.bot_right)

    def getPos(self):
        return self.top_left, self.bot_right


class Sobel:
    def __init__(self):
        self.kernels = []
        self.kernels.append((np.array([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]])))
        self.kernels.append((np.array([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]])))

    def filter(self, roi, axis=0):
        """
        :param roi: image that will have the filter applied to it
        :param axis: 0 is x, 1 is y, 3 is both
        :return:
        """

        if axis == 3:
            _filter = self.kernels[0] * roi
            sum_of_filter1 = _filter.sum()
            _filter = self.kernels[1] * roi
            sum_of_filter2 = _filter.sum()
            return ((sum_of_filter1 ** 2) + (sum_of_filter2 ** 2)) ** (1 / 2)
        else:
            _filter = self.kernels[axis] * roi
            return _filter.sum()

    def filterImage(self, image, window=None, axis=0):
        new_roi = []
        line = []
        if window is None:
            # go over entire image
            moving_kernel = Window(image, 3, 1)
        else:
            image = window.getImageInBoundary(image)
            moving_kernel = Window(image, 3, window.stride[0])

        new_tl, _ = moving_kernel.forwardPos()
        while moving_kernel.inBoundary(new_tl):
            roi = moving_kernel.getImageInBoundary(image)
            if moving_kernel.changedY():
                new_roi.append(line)
                line = []

            line.append(self.filter(roi, axis))

            moving_kernel.forwardMove()
            new_tl, _ = moving_kernel.forwardPos()

        return np.array(new_roi)


sobel = Sobel()

image = cv2.imread("1mb pic.png")
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

win = Window(image, 300, 1)
roi = win.getImageInBoundary(image)
cv2.imwrite("Output/roi_before_filter.png", roi, [cv2.IMWRITE_PNG_COMPRESSION, 0])
filtered = sobel.filterImage(image, axis=3)
cv2.imwrite("Output/image_after_filter.png", filtered, [cv2.IMWRITE_PNG_COMPRESSION, 0])
