

from Preprocess import split_train_val, create_datagen_oneband,MultiBandDataGenerator
from Unet_model import unet_three_band,unet_one_band,unet_two_band,unet_two_band_attention
from Loss import dice_loss,dice_coefficient,iou
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.keras.optimizers import Adadelta


def main():
    # 参数设置
    batch_size = 2
    target_size = (256, 256)
    val_size = 0.2
    epochs = 2
    modelsavedir= 'unet_threebands_val.hdf5'
    save_best_only = True
    image_folder = 'Data/test/imgs'  # 图像文件夹路径
    mask_folder = 'Data/test/masks'  # 掩模文件夹路径
    #image_folder = 'data/train/filter/set2_0.01/band_12'
    #mask_folder = 'data/train/filter/set2_0.01/labels'
    # 分割训练集和验证集
    train_images, train_masks, val_images, val_masks = split_train_val(image_folder, mask_folder, val_size=val_size)
    # 计算 steps_per_epoch 和 validation_steps
    steps_per_epoch = np.ceil(len(train_images) / batch_size)
    validation_steps = np.ceil(len(val_images) / batch_size)
    print( steps_per_epoch)
    print(validation_steps)
    # 根据选择的模式执行不同的代码
    mode = "ThreeBand"  # 或者 mode = "TwoBand"或者 "TwoBand_attention"


    # 定义模式对应的函数字典
    mode_functions = {


        "ThreeBand":{
            "create_datagen": MultiBandDataGenerator,
            "model": unet_three_band,
            "bands":3,
        },

        "TwoBand": {
            "create_datagen": MultiBandDataGenerator,
            "model": unet_two_band,
            "bands": 2,
        },
        "TwoBand_attention": {
            "create_datagen": MultiBandDataGenerator,
            "model": unet_two_band_attention,
            "bands": 2,
        }
    }
    # 获取对应模式的函数
    selected_mode = mode_functions.get(mode)
    if selected_mode:
        # 选择模式对应的函数
        create_datagen = selected_mode["create_datagen"]
        modelfunction=selected_mode["model"]
        bands=selected_mode["bands"]
        # 使用选择的函数
        train_gen = create_datagen(train_images, train_masks, batch_size, target_size,bands)
        val_gen = create_datagen(val_images, val_masks, batch_size, target_size,bands)
        model=modelfunction()
    else:
        print("Invalid mode selected.")


    print('Steps per epoch:', len(train_gen))
    for i, (images, masks) in enumerate(train_gen):
        print(f'Batch {i + 1}: images shape = {images.shape}, masks shape = {masks.shape}')
        if i >= 2:  # 仅示例，实际中不需要这个条件
            break


    # 模型编译
    model.compile(optimizer=Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-8),
                  loss=dice_loss,
                  metrics=[dice_coefficient, iou])

    # 定义模型检查点回调
    model_checkpoint = ModelCheckpoint(modelsavedir, monitor='val_loss', verbose=1, save_best_only=save_best_only)


    # 模型训练
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint],
        validation_data=val_gen,
        validation_steps=validation_steps
    )
    print(steps_per_epoch)
    print(validation_steps)

if __name__ == "__main__":
    main()























