import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
import rasterio  # 新添加，用于读取tif文件
import skimage.transform as trans

import glob
import os
from sklearn.model_selection import train_test_split

def preprocess_image_oneband(image):
    # 对数变换
    image = np.log1p(image)
    # 归一化到 [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def preprocess_image_twoband(image_path, target_size=(256, 256)):
    """针对双波段tif影像的预处理函数"""
    with rasterio.open(image_path) as src:
        image = src.read([1, 2])  # 读取前两个波段
        image = np.moveaxis(image, 0, -1)  # 重排轴，从(channels, height, width)到(height, width, channels)
        image = trans.resize(image, target_size)  # 调整大小
        image = np.log1p(image)  # 对数变换
        image = (image - image.min()) / (image.max() - image.min())  # 归一化到[0,1]
    return image


def preprocess_image_threeband_762(image_path, target_size=(256, 256)):
    """针对三波段tif影像的预处理函数，使用标准化"""
    with rasterio.open(image_path) as src:
        # 读取指定的三个波段
        image = src.read([7, 6, 2])
        image = np.moveaxis(image, 0, -1)  # 重排轴到(height, width, channels)
        image = trans.resize(image, target_size, preserve_range=True)  # 调整大小

        # 对每个波段进行标准化处理
        for i in range(image.shape[-1]):  # 遍历每个波段
            band = image[:, :, i]
            mean = band.mean()
            std = band.std()
            # 防止标准差为0的情况，如果为0则不做处理
            if std > 0:
                image[:, :, i] = (band - mean) / std
            else:
                image[:, :, i] = band - mean

    return image
def preprocess_image_tenband(image_path, target_size=(256, 256)):
    with rasterio.open(image_path) as src:
        # 读取前十个波段
        # 注意：根据您的影像数据，波段编号可能需要调整
        image = src.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        image = np.moveaxis(image, 0, -1)  # 重排轴到(height, width, channels)
        image = trans.resize(image, target_size, preserve_range=True)  # 调整大小

        # 对每个波段进行标准化处理
        for i in range(image.shape[-1]):  # 遍历每个波段
            band = image[:, :, i]
            mean = band.mean()
            std = band.std()
            # 防止标准差为0的情况，如果为0则不做处理
            if std > 0:
                image[:, :, i] = (band - mean) / std
            else:
                image[:, :, i] = band - mean

    return image

def preprocess_image_multiband(image_path, target_size=(256, 256)):
    with rasterio.open(image_path) as src:
        # 读取前十个波段
        # 注意：根据您的影像数据，波段编号可能需要调整
        image = src.read([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        image = np.moveaxis(image, 0, -1)  # 重排轴到(height, width, channels)
        image = trans.resize(image, target_size, preserve_range=True)  # 调整大小

        # 对每个波段进行标准化处理
        for i in range(image.shape[-1]):  # 遍历每个波段
            band = image[:, :, i]
            mean = band.mean()
            std = band.std()
            # 防止标准差为0的情况，如果为0则不做处理
            if std > 0:
                image[:, :, i] = (band - mean) / std
            else:
                image[:, :, i] = band - mean

    return image



def preprocess_mask(mask_path, target_size=(256, 256)):
    """掩码图像的预处理函数，不进行归一化"""
    mask = load_img(mask_path, color_mode='grayscale', target_size=target_size)
    mask = img_to_array(mask)
    return mask
# 这里省略了split_train_val和其它配置代码，因为它们不需要修改

import glob
import os


def get_corresponding_image(mask_path, image_paths):
    """
    根据掩码路径找到对应的图像路径。
    假设掩码文件名和图像文件名的唯一区别在于掩码文件名中有一个额外的 "Kumar-Roy" 字符串。
    """
    mask_basename = os.path.basename(mask_path)
    # 构建预期的图像文件名，通过删除 "Kumar-Roy" 来实现
    expected_image_basename = mask_basename.replace("_Kumar-Roy", "")

    # 在图像路径列表中查找匹配的图像文件
    for image_path in image_paths:
        image_basename = os.path.basename(image_path)
        if image_basename == expected_image_basename:
            return image_path
    return None  # 如果没有找到匹配的图像，返回 None
# LC08_L1GT_166058_20200816_20200816_01_RT_p00738
# LC08_L1GT_166058_20200816_20200816_01_RT_Kumar-Roy_p00738
def split_train_val(image_folder, mask_folder, val_size):
    image_paths = glob.glob(os.path.join(image_folder, '*.tif'))
    mask_paths = glob.glob(os.path.join(mask_folder, '*.tif'))

    # 匹配掩码和图像，而不是简单地排序
    matched_image_paths = []
    matched_mask_paths = []
    for mask_path in mask_paths:
        image_path = get_corresponding_image(mask_path, image_paths)
        if image_path:
            matched_image_paths.append(image_path)
            matched_mask_paths.append(mask_path)

    print(f"找到的匹配图像文件数量: {len(matched_image_paths)}")
    print(f"找到的匹配掩模文件数量: {len(matched_mask_paths)}")

    # 确保列表不为空
    if len(matched_image_paths) == 0 or len(matched_mask_paths) == 0:
        raise ValueError("未找到匹配的图像或掩模文件，请检查文件路径和匹配逻辑。")

    # 划分训练集和验证集
    train_images, val_images, train_masks, val_masks = train_test_split(
        matched_image_paths, matched_mask_paths, test_size=val_size, random_state=42
    )

    # 打印分割信息
    total = len(mask_paths)
    print(f"总共有 {total} 张掩模被分割。")
    print(f"训练集：{len(train_images)} 张图像，占比 {(len(train_images) / total):.2f}")
    print(f"验证集：{len(val_images)} 张图像，占比 {(len(val_images) / total):.2f}")
    return train_images, train_masks, val_images, val_masks


def split_train_val_test(image_folder, mask_folder, val_size, test_size):
    """
    分割数据为训练集、验证集和测试集，并打印出分割的信息。
    """
    image_paths = glob.glob(os.path.join(image_folder, '*.tif'))
    mask_paths = glob.glob(os.path.join(mask_folder, '*.tif'))

    # 匹配掩码和图像，而不是简单地排序
    matched_image_paths = []
    matched_mask_paths = []
    for mask_path in mask_paths:
        image_path = get_corresponding_image(mask_path, image_paths)
        if image_path:
            matched_image_paths.append(image_path)
            matched_mask_paths.append(mask_path)

    print(f"找到的匹配图像文件数量: {len(matched_image_paths)}")
    print(f"找到的匹配掩模文件数量: {len(matched_mask_paths)}")

    # 确保列表不为空
    if len(matched_image_paths) == 0 or len(matched_mask_paths) == 0:
        raise ValueError("未找到匹配的图像或掩模文件，请检查文件路径和匹配逻辑。")
    # 首先从全部数据中分割出测试集
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        matched_image_paths, matched_mask_paths, test_size=test_size, random_state=42
    )

    # 然后将剩余的数据分割为训练集和验证集
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, test_size=val_size / (1 - test_size), random_state=42
    )

    # 打印分割信息
    total = len(image_paths)
    print(f"总共有 {total} 张图像被分割。")
    print(f"训练集：{len(train_images)} 张图像，占比 {(len(train_images)/total):.2f}")
    print(f"验证集：{len(val_images)} 张图像，占比 {(len(val_images)/total):.2f}")
    print(f"测试集：{len(test_images)} 张图像，占比 {(len(test_images)/total):.2f}")
    print("分割成功！")

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def create_datagen_oneband(image_paths, mask_paths, batch_size, target_size,num_bands=1):
    # 将图像和掩码路径列表转换为 pandas DataFrame
    #image_df = pd.DataFrame({'filename': image_paths})
    #mask_df = pd.DataFrame({'filename': mask_paths})

    image_datagen = ImageDataGenerator(preprocessing_function=preprocess_image_oneband)
    mask_datagen = ImageDataGenerator()

    # 使用 DataFrame 代替列表
    image_generator = image_datagen.flow_from_dataframe(
        dataframe=image_df,
        x_col='filename',
        y_col=None,
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    mask_generator = mask_datagen.flow_from_dataframe(
        dataframe=mask_df,
        x_col='filename',
        y_col=None,
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    while True:
        img = next(image_generator)
        mask = next(mask_generator)
        yield img, mask

class MultiBandDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, target_size, num_bands):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_bands = num_bands  # 新增参数以指定波段数


    def __len__(self):
        return np.ceil(len(self.image_paths) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        batch_images = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.num_bands == 2:
            images = np.array([preprocess_image_twoband(image_path, self.target_size) for image_path in batch_images])
        elif self.num_bands == 3:
            images = np.array([preprocess_image_threeband_762(image_path, self.target_size) for image_path in batch_images])
        elif self.num_bands == 10:
            images = np.array([preprocess_image_tenband(image_path, self.target_size) for image_path in batch_images])

        else:
            images = np.array([preprocess_image_multiband(image_path, self.target_size) for image_path in batch_images])

        masks = np.array([preprocess_mask(mask_path, self.target_size) for mask_path in batch_masks])

        return images, masks
