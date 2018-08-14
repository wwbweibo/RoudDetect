"""
to do image segmentation using region grow method
"""
import numpy as np
from matplotlib import pyplot as plt
import DataTool

class RegionGrow:
    def __init__(self, image):
        """
        init class
        :param image: the image need to process
        """
        self.image = image
        self.imlist = []
        self.poslist = []
        self.retimg = np.zeros(shape=self.image.shape)
        self.imh, self.imw = image.shape
        self.block_w = 0
        self.block_h = 0

    def region_grow(self, mode=8):
        """
        image segmentation using region grow
        :type img: image
        :param img: input image
        :type mode: int
        :param mode: 4 or 8 only(8 as default)
        :return: new image after segmentation
        """
        for x in range(9):
            point = self.poslist[x]
            timblock = np.zeros(shape= self.image.shape)
            if point is None:
                continue
            # the position of the seed
            start_point = (point[0], point[1])
            # if the start point is nonzero, skip this point
            if self.retimg[start_point[0], start_point[1]] != 0:
                continue
            # the stack of point which need to be visited
            point_list = [start_point]
            # the dict of visited point
            visited_point = dict()
            visited_point[start_point] = start_point
            while len(point_list) > 0:
                # pop the top point and grow around this point
                point = point_list.pop()
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        # the point that is going to grow
                        new_point = point[0] + i, point[1] + j
                        # is the point visited, if visited pass the point
                        if visited_point.get(new_point) is not None:
                            continue
                        try:
                            if 0 <= new_point[0] < self.imh and 0 <= new_point[1] < self.imw and np.abs(
                                    self.image[new_point[0], new_point[1]] - self.image[start_point[0], start_point[1]]) < 25:
                                timblock[new_point[0], new_point[1]] = 255
                                point_list.append(new_point)
                                visited_point[new_point] = new_point
                        except:
                            print(new_point)
            self.im_merge2(timblock)
        self.retimg = self.retimg != 0
        return self.retimg

    def img_cut(self):
        """
        cut the image into 9 parts
        :return: list of image
        """
        # determine the size of pre block
        self.block_w = int(self.imw / 3)
        self.block_h = int(self.imh / 3)
        for i in range(3):
            for j in range(3):
                self.imlist.append(self.image[i * self.block_h:(i + 1) * self.block_h,
                                              j * self.block_w: (j + 1) * self.block_w])
        return self.imlist

    def min_pos(self):
        """
        to find out the darkness point in each block
        :return: a list of position in each block
        """
        min_val = np.min(np.min(self.image))
        block_index = 0
        for block in self.imlist:
            block = np.floor(block / 4)
            block = block.astype(np.uint8)
            posarr = np.where(block == min_val)
            # check is is this block contains min value
            if len(posarr[0]) <= 0:
                self.poslist.append(None)
                block_index += 1
                continue
            # todo using a more useful method to chose the seed
            # No.1 chose the point which is closest to center point
            # pick a point randomly and convert to global position
            center = (int(self.block_h/2),int(self.block_w / 2))    # center point
            pt = DataTool.min_distace(posarr, center)
            posw = int(block_index % 3) * self.block_w + pt[1]
            posh = int(block_index / 3) * self.block_h + pt[0]
            self.poslist.append((posh, posw))
            block_index += 1
        return self.poslist

    def im_merge2(self, temp_img):
        self.retimg = np.add(temp_img, self.retimg)
