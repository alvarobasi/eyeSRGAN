import cv2
import numpy as np
from tensorflow.python.client import device_lib
import os
import re
import matplotlib.pyplot as plt
from skimage import io


def print_available_devices():
    local_device_protos = [(x.name, x.device_type, x.physical_device_desc) for x in device_lib.list_local_devices()]
    for device_name, device_type, device_desc in local_device_protos:
        print("Device : {}\n\t type : {}\n\t desc :{}\n".format(device_name, device_type, device_desc))


def preprocess_HR(x):
    return np.divide(x.astype(np.float32), 127.5) - np.ones_like(x, dtype=np.float32)


def deprocess_HR(x):
    return np.clip((x + np.ones_like(x)) * 127.5, 0, 255)


def preprocess_LR(x):
    return np.divide(x.astype(np.float32), 255.)


def deprocess_LR(x):
    return np.clip(x * 255, 0, 255)


def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def _count_valid_files_in_directory(directory, white_list_formats):
    """Count files with extension in `white_list_formats` contained in a directory.
    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """

    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

    samples = 0
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples


def list_valid_filenames_in_directory(directory, white_list_formats):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        class_indices: dictionary mapping a class name to its index.
    # Returns
        filenames: the path of valid files in `directory`
    """

    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=False), key=lambda tpl: tpl[0])

    filenames = []
    basedir = os.path.dirname(directory)
    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                # add filename relative to directory
                absolute_path = os.path.join(root, fname)
                filenames.append(os.path.relpath(absolute_path, basedir))
    return filenames


# # Función importada de Gonzalo. Va iterando a través de las carpetas hasta recorrer los directorios enteros en
# busca de las imagenes. Devuelve la lista con todas las imagenes .png que cumplen con el patrón dado.
def get_list_of_files(folder):
    # Function to search folders and subfolders to obtain all png files
    files_in_folder = os.listdir(folder)
    output_files = []
    # Iterate over all the entries
    for entry in files_in_folder:
        # Create full path
        full_path = os.path.join(folder, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            output_files = output_files + get_list_of_files(full_path)
        elif re.match(r'\d{2}.png', entry):
            output_files.append(full_path)
    return output_files


def save_images(low_resolution_image, original_image, generated_image, path, data_format):
    if data_format == 'channels_first':
        low_resolution_image = np.transpose(low_resolution_image, (0, 3, 1, 2))
        original_image = np.transpose(original_image, (0, 3, 1, 2))
        generated_image = np.transpose(generated_image, (0, 3, 1, 2))

    io.imsave(path + '_orig_image.png', deprocess_HR(original_image).astype(np.uint8))
    io.imsave(path + '_gen_image.png', deprocess_HR(generated_image).astype(np.uint8))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 3, 1)
    # ax.imshow(deprocess_LR(low_resolution_image).astype(np.uint8))  # Cambio para LR de -1 a 1
    # ax.axis("off")
    # ax.set_title("Low-resolution")
    #
    # ax = fig.add_subplot(1, 3, 2)
    # ax.imshow(deprocess_HR(original_image).astype(np.uint8))
    # ax.axis("off")
    # ax.set_title("Original")
    #
    # ax = fig.add_subplot(1, 3, 3)
    # ax.imshow(deprocess_HR(generated_image).astype(np.uint8))
    # ax.axis("off")
    # ax.set_title("Generated")
    #
    # plt.savefig(path)
    #
    # plt.close()


def plot_images(index_list, images_hr, images_lr):
    for i in range(0, len(index_list), 2):
        plt.subplot(len(index_list), 2, i + 1)
        plt.axis('off')
        plt.imshow(images_hr[i])
        plt.subplots_adjust(wspace=0.5)
        plt.subplot(len(index_list), 2, i + 2)
        plt.axis('off')
        plt.imshow(images_lr[i])
        plt.subplots_adjust(wspace=0.5)

    plt.show()
