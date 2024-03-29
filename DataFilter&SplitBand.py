import os
import numpy as np
from PIL import Image
import shutil
import tifffile as tiff



def main():
    # 定义文件夹路径
    labels_dir = 'Data/africa/masks'
    output_dir_labels = 'Data/africa/filter_masks'
    images_dir =  'Data/africa/imgs'
    output_dir_images = 'Data/africa/filter_imgs'
    filter_value = 0.05
    filter_and_copy_images(labels_dir, output_dir_labels, images_dir, output_dir_images,filter_value)

def filter_and_copy_images(labels_dir, output_dir_labels, images_dir, output_dir_images, filter_value):
    if not os.path.exists(output_dir_labels):
        os.makedirs(output_dir_labels)

    if not os.path.exists(output_dir_images):
        os.makedirs(output_dir_images)

    # 为每个波段创建单独的文件夹
    # band_dirs = [os.path.join(output_dir_images, f'band_{i + 1}') for i in range(6)]
    # for band_dir in band_dirs:
    #     if not os.path.exists(band_dir):
    #         os.makedirs(band_dir)

    # 步骤1: 检测并筛选PNG图像，然后将符合条件的PNG图像复制到labels_filter目录
    i=0
    for filename in os.listdir(labels_dir):
        if filename.endswith('.tif'):
            image_path = os.path.join(labels_dir, filename)
            image = np.array(Image.open(image_path))
            pixelnumber=np.sum(image == 1)
            print(pixelnumber)
            print(image.size)
            ones_ratio = np.sum(image == 1) / image.size
            print(ones_ratio)
            if ones_ratio > filter_value:  # 1值比例大于0值
                print('value > 0.01')
                dest_path = os.path.join(output_dir_labels, filename)
                shutil.copy(image_path, dest_path)

                # 步骤2: 复制对应的TIF图像到image_filter目录
                # tif_filename = filename.replace('.png', '.tif')
                tif_filename = filename.replace("_Kumar-Roy", "")
                src_path = os.path.join(images_dir, tif_filename)
                if os.path.exists(src_path):
                    dest_tif_path = os.path.join(output_dir_images, tif_filename)
                    shutil.copy(src_path, dest_tif_path)
                    print("copy one tif")
                else:
                    i=i+1
                #     # 步骤3: 拆分TIF图像波段
                #     split_image_bands(src_path, band_dirs)


def split_image_bands(tif_path, band_dirs):
    img = tiff.imread(tif_path)
    filename = os.path.basename(tif_path)
    for i in range(6):  # 假设有6个波段
        band_data = img[:, :, i]
        band_path = os.path.join(band_dirs[i], filename)
        tiff.imwrite(band_path, band_data)
if __name__ == "__main__":
    main()

