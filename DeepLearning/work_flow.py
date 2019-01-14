import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from main_model import URD

# 加载模型
model, ae, encoder = URD('models/ae_0112.h5', 'DEC_model_final.h5').get_model()

im = cv.imread('../test_img/4.png', cv.IMREAD_GRAYSCALE)
im = cv.resize(im, (1024, 1024))

lst1 = []
lst1_resize = []
# 第一次图像分割
for i in range(16):
    for j in range(16):
        block = im[i*64:(i+1)*64 , j*64:(j+1)*64]
        lst1.append(block)
        lst1_resize.append(cv.resize(block, (16, 16)))

lst1_resize = np.asarray(lst1_resize)
lst1_resize = np.reshape(lst1_resize, (256, 16, 16, 1))

ret = model.predict(lst1_resize)
ret = ret.argmax(1)
idx_1 = np.where(ret == 0)[0]

lst2_origin = []
for x in idx_1:
    for i in range(4):
        for j in range(4):
            block = lst1[x][i*16:(i+1)*16 , j*16 : (j+1)*16]
            lst2_origin.append(block)

lst2 = np.asarray(lst2_origin)
lst2 = np.reshape(lst2, (len(idx_1) * 16 ,16,16,1))

ret = model.predict(lst2)
ret = ret.argmax(1)
idx = np.where(ret == 0)[0]

# 映射回原图
block_nos, in_block_nos = np.asarray(idx / 16, dtype=np.uint8), idx % 16

pos = []

for block_no, in_block_no in zip(block_nos, in_block_nos):
    row, col = int(idx_1[block_no] / 16), idx_1[block_no] % 16 
    i_row, i_col = int(in_block_no / 4), in_block_no % 4
    x = 64 * row + i_row * 16
    y = 64 * col + i_col * 16
    pos.append((x,y))
    im = cv.rectangle(im, (y, x), (y +16, x+16), 0, 2)

cv.imwrite('4.jpg', im)

# def hist_segmentation(hist, img):
#     """
#     do image segmentation using hist
#     :type img: gray image
#     :param img: origin image
#     :return: image after segmentation
#     """
#     max_index = np.where(hist == max(hist[1:]))
#     mask = hist[0:max_index[0][0]]
#     min_index = np.where(mask == min(mask))
#     ret, new_im = cv.threshold(img, min_index[0][0], 255, cv.THRESH_BINARY)
#     return new_im

# all_hist = np.zeros(shape=(256,1))
# for i in idx:
#     hist = cv.calcHist([lst2_origin[i]], [0], None, [256], [0, 255]) 
#     all_hist = all_hist + hist

# plt.plot(all_hist[1:])
# plt.show()
# for i in idx:
#     ret = hist_segmentation(all_hist, lst2[i])
#     cv.imshow('', ret)
#     cv.waitKey()