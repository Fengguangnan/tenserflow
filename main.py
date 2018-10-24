import os
import cv2
import numpy as np


def create_image_template(fname):
    img_row = 200
    img_col = 200
    num_images = 10
    num_line = 1

    # 全白
    img = np.zeros((img_row, img_col * num_images, 3), np.uint8)

    # 全黑
    img += 255

    while num_line < 10:
        x_pos = num_line * img_row
        cv2.line(img=img, pt1=(x_pos, 0), pt2=(x_pos, img_col), color=(0, 0, 0))
        num_line += 1

    cv2.imwrite(fname, img)
    return


def cut_image(fname, savepath, rows, cols):
    img = cv2.imread(fname)
    x_start = 0
    x_end = rows
    cnt = 0

    while x_start < cols:
        cut_img = img[0:rows, x_start:x_end]
        cv2.imwrite(os.path.join(savepath, str(cnt) + '.png'), cut_img)
        x_start += (rows + 1)
        x_end += rows
        cnt += 1
    return


# create_image_template('./test/template.png')
# cut_image('./test/template.png', './test', 200, 2000)


if __name__ == '__main__':
    str = 'a'
    print(ord(str))

