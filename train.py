

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from Loss import dice_loss, dice_coefficient, iou, f1_score
from Preprocess import split_train_val, MultiBandDataGenerator, split_train_val_test
from Unet_model import unet_three_band, unet_two_band, unet_two_band_attention
from ShowPerformance import plot_training_history
import os
from tensorflow.keras.optimizers import Adadelta, Adam, SGD  # 根据需要导入更多优化器

# 定义模式对应的函数字典
mode_functions = {

    "ThreeBand": {
        "bands": 3,
        "model": unet_three_band,
    },
    "TwoBand": {
        "bands": 2,
        "model": unet_two_band,
    },
    "TwoBand_attention": {
        "bands": 2,
        "model": unet_two_band_attention,
    }
}

def configure_and_compile_model(mode, optimizer_name, learning_rate, rho, epsilon, loss_name, metrics_names):
    # 在这里添加之前的优化器选择和编译模型的代码
    # 定义模式对应的函数字典
    selected_mode = mode_functions.get(mode)
    if selected_mode:
        # 选择模式对应的函数
        modelfunction=selected_mode["model"]
        model=modelfunction()
    else:
        print("Invalid mode selected.")
        return None
    # 动态选择优化器
    if optimizer_name == "Adadelta":
        optimizer = Adadelta(learning_rate=learning_rate, rho=rho, epsilon=epsilon)
    elif optimizer_name == "Adam":
        optimizer = Adam(learning_rate=learning_rate, epsilon=epsilon)
    elif optimizer_name == "SGD":
        optimizer = SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"未支持的优化器：{optimizer_name}")
    # 动态选择损失函数
    try:
        loss_function = eval(loss_name)
    except NameError:
        raise ValueError(f"未定义的损失函数：{loss_name}")
    # 动态选择评估指标
    metrics_functions = []
    for metric_name in metrics_names:
        try:
            metric_function = eval(metric_name)
            metrics_functions.append(metric_function)
        except NameError:
            raise ValueError(f"未定义的评估指标：{metric_name}")
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_functions)
    return model


def prepare_data_and_generators(mode,image_folder, mask_folder, val_size, test_size, perform_test, batch_size, target_size):
    """
    准备数据并创建数据生成器。

    参数:
    - image_folder: 图像文件夹路径。
    - mask_folder: 掩模文件夹路径。
    - val_size: 验证集大小。
    - test_size: 测试集大小。
    - perform_test: 是否进行测试集分割和评估。
    - batch_size: 批次大小。
    - target_size: 目标图像尺寸。
    - bands: 使用的波段数量。

    返回:
    - train_gen: 训练数据生成器。
    - val_gen: 验证数据生成器。
    - test_gen: 测试数据生成器，如果 perform_test 为 False，则返回 None。
    """
    if mode in mode_functions:
        bands = mode_functions[mode]["bands"]
        # 根据bands配置模型...
        print(f"配置模型，使用波段数：{bands}")
    else:
        print("未知的模式配置。")
        raise ValueError(f"未知的模式配置：{mode}")

    # 选择模式对应的函数
    create_datagen = MultiBandDataGenerator
    # 使用之前描述的分割函数
    if perform_test:
        train_images, train_masks, val_images, val_masks, test_images, test_masks = split_train_val_test(
            image_folder, mask_folder, val_size=val_size, test_size=test_size
        )
    else:
        train_images, train_masks, val_images, val_masks = split_train_val(image_folder, mask_folder, val_size=val_size)
        test_images, test_masks = [], []  # 没有测试集时返回空列表

    # 创建数据生成器实例
    train_gen = create_datagen(train_images, train_masks, batch_size, target_size, bands)
    val_gen = create_datagen(val_images, val_masks, batch_size, target_size, bands)

    if perform_test:
        test_gen = create_datagen(test_images, test_masks, batch_size, target_size, bands)
    else:
        test_gen = None  # 没有测试集时返回 None

    return train_gen, val_gen, test_gen


def main():
    # 参数设置
    batch_size = 16
    target_size = (256, 256)
    val_size = 0.2
    epochs = 40
    save_best_only = True
    image_folder = 'Data/africa/imgs'  # 图像文件夹路径
    mask_folder = 'Data/africa/masks'  # 掩模文件夹路径
    perform_test = True  # 可以根据需要设置为 False 来禁用测试集的分割和评估
    test_size= 0.3
    plot_loss= True
    # 基础配置
    mode = "ThreeBand"
    optimizer_name = "Adadelta"
    learning_rate = 1.0
    rho = 0.95
    epsilon = 1e-8
    loss_name = "dice_loss"
    metrics_names = ["dice_coefficient", "iou", "f1_score"]  # 将指标列表转换为字符串以便使用
    metrics_str = "_".join(metrics_names)

    # 创建保存目录的基本名称部分
    base_save_name = f"{mode}_opt-{optimizer_name}_lr-{learning_rate}_rho-{rho}_eps-{epsilon}_loss-{loss_name}_metrics-{metrics_str}"
    # 设置模型保存目录和文件名
    modelsavedir = f'models/{base_save_name}.hdf5'
    # 确保目录存在
    os.makedirs(os.path.dirname(modelsavedir), exist_ok=True)
    print("模型将被保存到:", modelsavedir)



    train_gen, val_gen, test_gen=prepare_data_and_generators(mode,image_folder, mask_folder, val_size, test_size, perform_test, batch_size, target_size)
    #模型编译
    model=configure_and_compile_model(mode=mode, optimizer_name=optimizer_name, learning_rate=learning_rate, rho=rho, epsilon=epsilon, loss_name=loss_name, metrics_names=metrics_names)
    # 定义模型检查点回调
    model_checkpoint = ModelCheckpoint(modelsavedir, monitor='val_loss', verbose=1, save_best_only=save_best_only)
    #模型训练
    history=train_model(model=model, train_gen=train_gen, val_gen=val_gen, epochs=epochs, model_checkpoint=model_checkpoint)
    # 假设所有之前的步骤都已完成
    # 在训练完成后评估和保存结果
    evaluate_and_save_results(model,test_gen, base_save_name, history, plot_loss)


def train_model(model, train_gen, val_gen, epochs, model_checkpoint):
    # 在这里添加之前模型训练的代码
    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)
    print(f"训练每个 epoch 的步骤数 (steps_per_epoch): {steps_per_epoch}")
    print(f"验证每个 epoch 的步骤数 (validation_steps): {validation_steps}")
    # 模型训练
    history=model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint],
        validation_data=val_gen,
        validation_steps=validation_steps
    )
    return history
def evaluate_and_save_results(model, test_gen, base_save_name, history, plot_loss):
    """
    评估模型性能，并保存模型和训练结果图表。

    参数:
    - model: 训练完成的模型。
    - test_gen: 测试数据生成器。
    - base_save_name: 基础保存文件名，用于区分不同的训练配置。
    - history: 训练历史对象。
    - plot_loss: 是否绘制和保存损失变化图表。
    """

    # 测试集评估
    if test_gen is not None:
        test_steps = len(test_gen)
        test_loss, test_dice, test_iou, test_f1 = model.evaluate(test_gen, steps=test_steps)
        print(f"测试集上的损失: {test_loss}")
        print(f"测试集上的 Dice 系数: {test_dice}")
        print(f"测试集上的 IoU: {test_iou}")
        print(f"测试集上的 F1: {test_f1}")
        # 在模型保存路径中加入测试损失和指标
        model_save_path = f"{base_save_name}_testloss-{test_loss:.4f}_dice-{test_dice:.4f}_iou-{test_iou:.4f}_f1-{test_f1:.4f}.hdf5"
    else:
        # 如果没有进行测试评估，则直接使用基础路径
        model_save_path = base_save_name
    # 保存模型
    model.save(model_save_path)
    print(f"模型已保存到: {model_save_path}")
    # 绘制和保存训练损失及指标变化图表
    if plot_loss:
        plot_loss_save_dir = f'plots/{base_save_name}'
        os.makedirs(os.path.dirname(plot_loss_save_dir), exist_ok=True)
        print("训练损失和指标绘图将被保存到:", plot_loss_save_dir)
        plot_training_history(history, additional_metrics=['dice_coefficient', 'iou', 'f1_score'],
                              save_path=os.path.join(plot_loss_save_dir, '_training_plot.png'))
if __name__ == "__main__":
    main()























