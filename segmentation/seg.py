import math

import cv2
from matplotlib import pyplot as plt
import numpy as np


def load_image(img_path: str):
    return cv2.imread(img_path)


def bgr_2_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bgr_2_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rm_green_blue(img):
    img[:, :, 0] = 0  # blue
    img[:, :, 1] = 0  # green
    return img


def save_plt(plt_obj, f_name):
    plt.figure(figsize=(10, 10))
    plt.imsave(f'static/output/{f_name}', plt_obj)
    plt.close()


def save_multi_plt(plt_objs, name):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 20))
    ax1.imshow(plt_objs[0])
    ax2.imshow(plt_objs[1])
    ax3.imshow(plt_objs[2])
    plt.savefig(f'static/output/{name}', bbox_inches = 'tight')
    plt.close()


def apply_mask(img, mask_bgr_min, mask_bgr_max, morph_open_iter, morph_close_iter):

    """
    :param img: Grey version
    :param mask_min: BGR Values where each element in the list corresponds to a color
    :param mask_max: BGR Values where each element in the list corresponds to a color
    :param morph_open_iteration: Higher iteration eliminates small points
    :param morph_close_iteration: Higher iteration combines neighboring cells into one
    :return:
    """

    mask = cv2.inRange(img, np.array(mask_bgr_min), np.array(mask_bgr_max))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_open_iter)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)
    return kernel, opening, close


def count_cells(img, close, min_area, put_count: bool):
    """
    :param
    :param
    :param
    """
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cells = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            cv2.drawContours(img, [c], -1, (0, 255, 255), 2)  # bgr values
            cells += 1
            if put_count:
                x = c[0][0][0]
                y = c[0][0][1]
                cv2.putText(img, f"{cells}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (194, 255, 0), 2)

    return img, cells


def run_segmenter(img_path, mask_bgr_min=(0, 0, 50), mask_bgr_max=(0, 0, 255), morph_open_iter=1, morph_close_iter=5,
                  min_area=200, w_count=True,):

    orig_img = load_image(img_path)
    orig_img_cpy = orig_img.copy()
    img_bgr = orig_img.copy()
    img_red = rm_green_blue(img_bgr)
    kernel, img_opening, img_close = apply_mask(img=img_red, mask_bgr_min=mask_bgr_min, mask_bgr_max=mask_bgr_max,
                                                morph_open_iter=morph_open_iter, morph_close_iter=morph_close_iter)

    img_out, cell_count = count_cells(img=orig_img_cpy, close=img_close, min_area=min_area, put_count=w_count)
    # save_plt(bgr_2_rgb(orig_img), 'input.png')
    # save_plt(img_close, 'mask.png')
    # save_plt(bgr_2_rgb(img_out), 'output.png')
    # save_multi_plt((bgr_2_rgb(orig_img), img_close, bgr_2_rgb(img_out)), 'output.png')
    return bgr_2_rgb(img_out)


def run_merged_segment(img_path, mask_bgr_min=(0, 0, 50), mask_bgr_max=(0, 0, 255), morph_open_iter=1,
                       morph_close_iter=5, min_area=150, w_count=True,):

    orig_img = load_image(img_path)
    orig_img_cpy = orig_img.copy()
    img_red_n_green = orig_img.copy()
    kernel, img_opening, img_close = apply_mask(img=img_red_n_green, mask_bgr_min=mask_bgr_min,
                                                mask_bgr_max=mask_bgr_max, morph_open_iter=morph_open_iter,
                                                morph_close_iter=morph_close_iter)

    img_out, cell_count = count_cells(img=orig_img_cpy, close=img_close, min_area=min_area, put_count=w_count)

    return cell_count, bgr_2_rgb(orig_img), bgr_2_rgb(img_out)