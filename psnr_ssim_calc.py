import tensorflow as tf
import cv2
import os
import math
import numpy as np

if __name__ == "__main__":

    orig_path = './datasets/A_guadiana_final/User_01/Grid_15/HP_001/01_padded.png'
    bic_path = './datasets/A_guadiana_final/User_01/Grid_15/HP_001/01_padded_bicubic.png'
    srgan_path = './datasets/A_guadiana_final/User_01/Grid_15/HP_001/01_padded_gen.png'
    mse_path = './datasets/A_guadiana_final/User_01/Grid_15/HP_001/01_padded_mse.png'

    list_file_path = './outputs/listado_imagenes.npy'
    if os.path.isfile(list_file_path):
        list_files = np.load(list_file_path)

    np.random.shuffle(list_files)

    list_files_sample = np.random.choice(list_files, size=30, replace=False)

    list_file_mse = []
    list_file_gen = []
    list_file_bic = []
    for i, file in enumerate(list_files_sample):
        base_name = os.path.splitext(file)[0]
        output_path_mse = f'{base_name}_mse.png'
        output_path_gen = f'{base_name}_gen.png'
        output_path_bic = f'{base_name}_bicubic.png'

        list_file_bic.append(output_path_bic)
        list_file_gen.append(output_path_gen)
        list_file_mse.append(output_path_mse)

    psnr_bic_array = []
    psnr_gen_array = []
    psnr_mse_array = []

    ssim_bic_array = []
    ssim_gen_array = []
    ssim_mse_array = []
    for i, file in enumerate(list_files_sample):
        orig = tf.image.decode_png(tf.io.read_file(list_files_sample[i]))
        bic = tf.image.decode_png(tf.io.read_file(list_file_bic[i]))
        srgan = tf.image.decode_png(tf.io.read_file(list_file_gen[i]))
        mse = tf.image.decode_png(tf.io.read_file(list_file_mse[i]))

        psnr_bic_array.append(tf.image.psnr(bic, orig, max_val=255).numpy())
        psnr_mse_array.append(tf.image.psnr(mse, orig, max_val=255).numpy())
        psnr_gen_array.append(tf.image.psnr(srgan, orig, max_val=255).numpy())

        ssim_bic_array.append(tf.image.ssim(bic, orig, max_val=255).numpy())
        ssim_gen_array.append(tf.image.ssim(srgan, orig, max_val=255).numpy())
        ssim_mse_array.append(tf.image.ssim(mse, orig, max_val=255).numpy())

    print("\n---- BICUBIC ----")
    print("--> PSNR(dB):")
    print("   - mean: ", np.mean(psnr_bic_array))
    print("   - std: ", np.std(psnr_bic_array))
    print("--> SSIM: ")
    print("   - mean: ", np.mean(ssim_bic_array))
    print("   - std: ", np.std(ssim_bic_array))

    print("\n---- SRGAN ----")
    print("--> PSNR(dB):")
    print("   - mean: ", np.mean(psnr_gen_array))
    print("   - std: ", np.std(psnr_gen_array))
    print("--> SSIM: ")
    print("   - mean: ", np.mean(ssim_gen_array))
    print("   - std: ", np.std(ssim_gen_array))

    print("\n---- MSE ----")
    print("--> PSNR(dB):")
    print("   - mean: ", np.mean(psnr_mse_array))
    print("   - std: ", np.std(psnr_mse_array))
    print("--> SSIM: ")
    print("   - mean: ", np.mean(ssim_mse_array))
    print("   - std: ", np.std(ssim_mse_array))

    print("List of samples used:")
    print(list_files_sample)
