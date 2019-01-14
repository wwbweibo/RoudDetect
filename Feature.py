import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def get_area_pos(img, filter_size=1000, flag=False):
    """
    从图形中获取区域面积及位置

    :param img: 输入图形
    :param filter_size: 过滤的面积大小
    :param flag: show result?
    :return: list(area,pos);area:int, pos（x,y,w,h）
    """

    # 检查类型
    if img.dtype is not np.uint8:
        img = img.astype(np.uint8)
    # 获取边缘点
    image, contours, hierarchy = cv.findContours(img,
                                                 cv.RETR_TREE,
                                                 cv.CHAIN_APPROX_NONE)
    result_list = []
    # 统计面积以及位置
    for con in contours:
        image = cv.drawContours(image, con, -1, 255)
        area = cv.contourArea(con)
        if area > filter_size:
            x, y, w, h = cv.boundingRect(con)

            result_list.append((area, x, y, w, h))
            if w / h > 1:
                print("横向裂缝")
                print(find_min_max_width(con))
            else:
                print("纵向裂缝")
                print(find_min_max_width_vertical(con))
            if flag:
                temp_img = np.zeros(image.shape)
                temp_img = cv.drawContours(temp_img, con, -1, 255)
                print('x:%d，y:%d，w:%d，h:%d' % (x, y, w, h))
                temp_img = cv.rectangle(temp_img, (x, y), (x + w, y + h), 180)
                cv.imshow("result", temp_img)
                cv.waitKey()
    return result_list


def connected_region_label(img, flag=False):
    """
    对二值图像进行连通性分析

    :param img: 输入图像
    :param flag: 是否显示结果
    :return: 连通区域总数，标记的每个连通区域
    """

    img = img.astype(np.uint8)
    result, labels = cv.connectedComponents(img)
    if flag:
        temp = labels * 10
        plt.imshow(temp, cmap="gray")
        plt.show()
        print(result)

    labels_imgs = []
    for i in range(result):
        ret = np.asarray((labels == i), np.bool)
        labels_imgs.append(ret)

    return result, labels_imgs


def find_min_max_width(contour):
    min_dict = dict()
    max_dict = dict()
    max_width = 0
    min_width = 999
    min_pos = 0
    max_pos = 0
    for pt in contour:
        if min_dict.get(pt[0][0]) is None or max_dict.get(pt[0][0]) is None:
            min_dict[pt[0][0]] = max_dict[pt[0][0]] = pt[0][1]
        elif min_dict[pt[0][0]] > pt[0][1]:
            min_dict[pt[0][0]] = pt[0][1]
            if max_width < (max_dict[pt[0][0]] - min_dict[pt[0][0]]):
                max_width = max_dict[pt[0][0]] - min_dict[pt[0][0]]
                max_pos = pt[0][0]
            if min_width > (max_dict[pt[0][0]] - min_dict[pt[0][0]]):
                min_width = max_dict[pt[0][0]] - min_dict[pt[0][0]]
                min_pos = pt[0][0]
        elif max_dict[pt[0][0]] < pt[0][1]:
            max_dict[pt[0][0]] = pt[0][1]
            if max_width < (max_dict[pt[0][0]] - min_dict[pt[0][0]]):
                max_width = max_dict[pt[0][0]] - min_dict[pt[0][0]]
                max_pos = pt[0][0]
            if min_width > (max_dict[pt[0][0]] - min_dict[pt[0][0]]):
                min_width = max_dict[pt[0][0]] - min_dict[pt[0][0]]
                min_pos = pt[0][0]
    return max_width, min_width


def find_min_max_width_vertical(contour):
    min_dict = dict()
    max_dict = dict()
    max_width = 0
    min_width = 999
    min_pos = 0
    max_pos = 0
    for pt in contour:
        if min_dict.get(pt[0][1]) is None or max_dict.get(pt[0][1]) is None:
            min_dict[pt[0][1]] = max_dict[pt[0][1]] = pt[0][0]
        elif min_dict[pt[0][1]] > pt[0][0]:
            min_dict[pt[0][1]] = pt[0][0]
            if max_width < (max_dict[pt[0][1]] - min_dict[pt[0][1]]):
                max_width = max_dict[pt[0][1]] - min_dict[pt[0][1]]
                max_pos = pt[0][0]
            if min_width > (max_dict[pt[0][1]] - min_dict[pt[0][1]]):
                min_width = max_dict[pt[0][1]] - min_dict[pt[0][1]]
                min_pos = pt[0][0]
        elif max_dict[pt[0][1]] < pt[0][0]:
            max_dict[pt[0][1]] = pt[0][0]
            if max_width < (max_dict[pt[0][1]] - min_dict[pt[0][1]]):
                max_width = max_dict[pt[0][1]] - min_dict[pt[0][1]]
                max_pos = pt[0][0]
            if min_width > (max_dict[pt[0][1]] - min_dict[pt[0][1]]):
                min_width = max_dict[pt[0][1]] - min_dict[pt[0][1]]
                min_pos = pt[0][0]
    return max_width, min_width
