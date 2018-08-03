import numpy as np
import math

def distance_calc(point1, point2):
    """
    to calculate the distance of 2 points
    :param point1: pt1
    :param point2: pt2
    :return: the distance
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def min_distace(pointlist, center):
    """
    given a group of points and a center point to find out the point which is closest to center point
    :type pointlist: tuple
    :param pointlist: the list of point
    :type center: tuple
    :param center: center point
    :return: the point closest to center
    """
    dis_dict = dict()
    dis_list = []

    # for all the point calculate distance
    for index in range(len(pointlist[0])):
        pt = (pointlist[0][index],pointlist[1][index])
        dis = distance_calc(pt, center)
        dis_dict[dis] = pt
        dis_list.append(dis)
    dis_list.sort()
    return dis_dict[dis_list[0]]
