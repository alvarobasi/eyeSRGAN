import os
import re
import numpy as np
import json
import cv2 as cv
import matplotlib.pyplot as plt
import utils
import multiprocessing as mp


def getData(path, convert_to_np=False):
    with open(path, 'r') as handle:
        data_dict = json.load(handle)
    return data_dict


def files_same_distance(file_name, distance=0.4, error=0.005):
    DISTANCE_ABS = distance * (63.9664 - 35.8273) + 35.8273
    ERROR = error * (63.9664 - 35.8273)
    json_file = getData(file_name)
    position = json_file['Position']
    if DISTANCE_ABS - ERROR <= position[0][2] <= DISTANCE_ABS + ERROR:
        return True


def get_list_of_files(folder, string_to_search=r'\d\d.json'):
    files_in_folder = os.listdir(folder)
    output_files = []
    for entry in files_in_folder:
        full_path = os.path.join(folder, entry)
        if os.path.isdir(full_path):
            output_files = output_files + get_list_of_files(full_path, string_to_search=string_to_search)
        elif re.match(string_to_search, entry):
            output_files.append(full_path)
    return output_files


def get_padding(file_name):
    json_file = getData(file_name)
    padding = json_file['Values_of_Padding_t_b_l_r']
    return padding[0]


def img_proc(file_name):
    max_height = 84
    max_width = 388

    img = cv.imread(file_name, cv.IMREAD_COLOR)
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(gray_img, 1, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    X, Y, W, H = cv.boundingRect(contour)

    cropped = img[(Y + 1):(Y - 1) + H, (X + 1):(X - 1) + W]
    h, w, d = cropped.shape

    top = (max_height - h) // 2
    bot = (max_height - h) - top

    left = (max_width - w) // 2
    right = (max_width - w) - left

    # result = cv.copyMakeBorder(cropped, top, bot, left, right, cv.BORDER_REPLICATE)
    value = [255, 0, 0]
    result = cv.copyMakeBorder(cropped, top, bot, left, right, cv.BORDER_CONSTANT, None, value)

    base_name = os.path.splitext(file_name)[0]
    output_path = f'{base_name}_padded_green.png'
    cv.imwrite(output_path, result)


if __name__ == "__main__":

    dataset_path = './datasets/A_guadiana_final/'

    list_file_path = './outputs/listado_imagenes.npy'
    if os.path.isfile(list_file_path):
        list_files = np.load(list_file_path)
    else:
        list_files = utils.get_list_of_files(dataset_path)
        np.save(list_file_path, list_files)

    # Use multiprocessing to speed up the image padding
    pool = mp.Pool(mp.cpu_count())
    pool.map(img_proc, list_files)
    pool.close()
    pool.join()
