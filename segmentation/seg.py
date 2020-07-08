import math

import cv2
import numpy as np


def load_image(img_path: str):
    return cv2.imread(img_path)


def bgr_2_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bgr_2_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_mask(img, mask_bgr_min: list, mask_bgr_max: list, morph_open_iter, morph_close_iter):

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

    img = bgr_2_rgb(img)
    return img, cells


def run_segmenter(img_path, mask_bgr_min, mask_bgr_max, morph_open_iter, morph_close_iter, min_area, put_count,):

    orig_img = load_image(img_path)
    img_rgb = bgr_2_rgb(orig_img)
    img_gray = bgr_2_gray(orig_img)
    kernel, img_opening, img_close = apply_mask(img=img_gray, mask_bgr_min=mask_bgr_min, mask_bgr_max=mask_bgr_max,
                                                morph_open_iter=morph_close_iter, morph_close_iter=morph_close_iter)

    img_out, cell_count = count_cells(img=img_rgb, close=img_close, min_area=min_area, put_count=put_count)

    return img_out, cell_count
