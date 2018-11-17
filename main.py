# coding:utf-8
import time
import Feature
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


import PreProcess
import RegionGrow

if __name__ == "__main__":

    origin = PreProcess.read_image("test_img/ceshi1.jpg", color_code=cv.IMREAD_ANYCOLOR)
    origin = PreProcess.resize_img(origin)
    print(origin.shape)
    img = PreProcess.convert_color(origin)
    if img is not None:
        img = PreProcess.equalize_hist(img, flag=False)
        img = PreProcess.center_avg_imp(img, ksize=20, flag=False)
        img = PreProcess.med_blur(img, ksize=5, flag=False)
        start_time = time.time()
        rg = RegionGrow.RegionGrow(img)
        rg.img_cut()
        rg.min_pos()
        img = rg.region_grow()

        end_time = time.time()
        print("run in %.2f" % (end_time - start_time))
        # img = rg.im_merge()
        plt.imshow(img, cmap="gray")
        plt.show()
        # img = PreProcess.binary_image(img, 100, True)
        img = PreProcess.med_blur(img, ksize=3, flag=False)
        plt.imsave("a.jpg",img,cmap='gray')
        # img = cv.dilate(img, np.array([[1,1,1],[1,1,1],[1,1,1]]))
        result, imgs = Feature.connected_region_label(img, flag=False)
        for img in imgs[1:]:
            area_result = Feature.get_area_pos(img, flag=False)
            for r in area_result:
                origin = cv.rectangle(origin,
                                      (r[1], r[2]),
                                      (r[1] + r[3], r[2] + r[4]),
                                      (0, 0, 0),
                                      thickness=2)
                origin = cv.putText(origin,
                                    'area:' + str(r[0]),
                                    color=(0, 0, 255),
                                    org=(r[1], r[2] + 30),
                                    fontFace=cv.FONT_ITALIC,
                                    fontScale=0.5)
        origin = PreProcess.convert_color(origin, cv.COLOR_BGR2RGB)
        plt.imshow(origin)
        plt.title("识别结果")
        plt.show()
        plt.imsave( "b.jpg",origin)
