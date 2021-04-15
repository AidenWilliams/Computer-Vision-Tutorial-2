import numpy as np


class Window(object):
    def __init__(self, image, n, s):
        self.x_boundary = image.shape[1] + n
        self.y_boundary = image.shape[0] + n
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

    def getPos(self):
        return self.top_left, self.bot_right

    def forwardPos(self):
        # Case when you need to go down and start new line
        if (self.bot_right + self.stride)[0] >= (self.x_boundary - self.height):
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

    def getImageInBoundary(self, image):
        new_image = []
        for i in range(self.top_left[1], self.bot_right[1]):
            if i >= image.shape[0]:
                continue
            new_image.append(image[i][self.top_left[0]: self.bot_right[0]])

        if self.channels == 1:
            return np.resize(np.array(new_image), (self.height, self.height))
        else:
            return np.resize(np.array(new_image), (self.height, self.height, self.channels))

    def __str__(self):
        return "Top Left Corner " + str(self.top_left) + "\nBot Right Corner " + str(self.bot_right)

# make all kernels same with transpose for [1]
class Sobel:
    def __init__(self, weight=1/8):
        self.kernels = []
        self.kernels.append((np.array([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]])))
        self.kernels.append((np.array([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]])))
        self.weight = weight

    def filter(self, roi, axis=2, channels=1):
        if axis == 2:
            ret = []
            for i in range(channels):
                if channels != 1:
                    _filter = self.kernels[0] * roi[:, :, i]
                else:
                    _filter = self.kernels[0] * roi
                sum_of_filter1 = _filter.sum()
                if channels != 1:
                    _filter = self.kernels[1] * roi[:, :, i]
                else:
                    _filter = self.kernels[1] * roi
                sum_of_filter2 = _filter.sum()
                ret.append((((sum_of_filter1 ** 2) + (sum_of_filter2 ** 2)) ** (1 / 2)) * self.weight)

            return np.array(ret)
        else:
            ret = []
            for i in range(channels):
                if channels != 1:
                    _filter = self.kernels[axis] * roi[:, :, i]
                else:
                    _filter = self.kernels[axis] * roi
                ret.append(_filter.sum() * self.weight)

            return np.array(ret)

    def filterImage(self, image, stride=1, window=None, axis=2):
        new_roi = []
        line = []
        if window is None:
            moving_kernel = Window(image, 3, stride)
        else:
            image = window.getImageInBoundary(image)
            moving_kernel = Window(image, 3, stride)

        new_tl, _ = moving_kernel.forwardPos()
        while moving_kernel.inBoundary(new_tl):
            roi = moving_kernel.getImageInBoundary(image)
            if moving_kernel.changedY():
                new_roi.append(line)
                line = []

            line.append(self.filter(roi, axis, moving_kernel.channels))

            moving_kernel.forwardMove()
            new_tl, _ = moving_kernel.forwardPos()

        return np.array(new_roi)


class Kernel:
    def __init__(self, kernel, weight):
        self.kernel = kernel
        self.weight = weight

    def filter(self, roi, channels=1):
        ret = []
        for i in range(channels):
            if channels != 1:
                _filter = self.kernel * roi[:, :, i]
            else:
                _filter = self.kernel * roi
            ret.append(_filter.sum() * self.weight)

        return np.array(ret)

    def filterImage(self, image, stride=1, window=None):
        new_roi = []
        line = []
        if window is None:
            # go over entire image
            moving_kernel = Window(image, self.kernel.shape[0], stride)
        else:
            image = window.getImageInBoundary(image)
            moving_kernel = Window(image, self.kernel.shape[0], stride)

        new_tl, _ = moving_kernel.forwardPos()
        while moving_kernel.inBoundary(new_tl):
            roi = moving_kernel.getImageInBoundary(image)
            if moving_kernel.changedY():
                new_roi.append(line)
                line = []

            line.append(self.filter(roi, moving_kernel.channels))

            moving_kernel.forwardMove()
            new_tl, _ = moving_kernel.forwardPos()

        return np.array(new_roi)


class Gaussian:
    def __init__(self, size, weight):
        # Source: #https://gist.github.com/andrewgiessel/4635563
        fwhm = size // 2
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        self.kernel = Kernel(np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2), weight)

    def filterImage(self, image, stride=1, window=None):
        return self.kernel.filterImage(image, stride, window)


class Bilinear:
    def __init__(self, weight):
        self.kernel = Kernel(np.array([[1, 2, 1],
                                       [2, 4, 2],
                                       [1, 2, 1]]),
                             weight)

    def filterImage(self, image, stride=1, window=None):
        return self.kernel.filterImage(image, stride, window)
