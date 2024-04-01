

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from Loss import dice_loss, dice_coefficient, iou, f1_score
from Preprocess import split_train_val, MultiBandDataGenerator, split_train_val_test
from Unet_model import unet_three_band, unet_two_band, unet_two_band_attention,unet_three_band_attention,unet_ten_band
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
    },
    "ThreeBand_attention":{
        "bands": 3,
        "model": unet_three_band_attention,
    },
    "TenBand":{
        "bands": 10,
        "model": unet_ten_band,
    }
}

config = {
    "batch_size": 2,
    "target_size": (256, 256),
    "val_size": 0.2,
    "epochs": 5,
    "save_best_only": True,
    "image_folder": 'Data/test/imgs/imgs',  # 图像文件夹路径
    "mask_folder": 'Data/test/masks/masks',  # 掩模文件夹路径
    "perform_test": True,  # 是否进行测试集分割和评估
    "test_size": 0.3,
    "plot_loss": True,

    "mode": "ThreeBand",
    "optimizer_name": "Adadelta",
    "learning_rate": 1.0,
    "rho": 0.95,
    "epsilon": 1e-8,
    "loss_name": "dice_loss",
    "metrics_names": ["dice_coefficient", "iou", "f1_score"]
}

def main():
    # 参数设置字典
    base_save_name = f"{config['mode']}_opt-{config['optimizer_name']}_lr-{config['learning_rate']}_rho-{config['rho']}_eps-{config['epsilon']}"
    modelsavedir = f'models/{base_save_name}.hdf5'
    os.makedirs(os.path.dirname(modelsavedir), exist_ok=True)
    print("模型将被保存到:", modelsavedir)

    # 从字典中传递参数
    train_gen, val_gen, test_gen = prepare_data_and_generators(
        config["mode"],
        config["image_folder"],
        config["mask_folder"],
        config["val_size"],
        config["test_size"],
        config["perform_test"],
        config["batch_size"],
        config["target_size"]
    )

    # 模型编译
    model = configure_and_compile_model(
        mode=config["mode"],
        optimizer_name=config["optimizer_name"],
        learning_rate=config["learning_rate"],
        rho=config["rho"],
        epsilon=config["epsilon"],
        loss_name=config["loss_name"],
        metrics_names=config["metrics_names"]
    )

    # 定义模型检查点回调
    model_checkpoint = ModelCheckpoint(modelsavedir, monitor='val_loss', verbose=1,
                                       save_best_only=config["save_best_only"])

    # 模型训练
    history = train_model(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=config["epochs"],
        model_checkpoint=model_checkpoint
    )

    # 评估和保存结果
    evaluate_and_save_results(
        model,
        test_gen,
        base_save_name,
        history,
        config["plot_loss"]
    )


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
        model_save_path = f"models/{base_save_name}_testloss-{test_loss:.4f}_dice-{test_dice:.4f}_iou-{test_iou:.4f}_f1-{test_f1:.4f}.hdf5"
        test_results = {
            "Test Loss": test_loss,
            "Dice Coefficient": test_dice,
            "IoU": test_iou,
            "F1 Score": test_f1
        }

        # 保存配置参数和测试结果到文本文件
        save_results_to_txt(config, test_results, file_path='training_results.txt')
    else:
        # 如果没有进行测试评估，则直接使用基础路径
        model_save_path = f"models/{base_save_name}.hdf5"
    # 保存模型
    model.save(model_save_path)
    print(f"模型已保存到: {model_save_path}")
    # 绘制和保存训练损失及指标变化图表
    if plot_loss:
        # 定义基础目录
        plot_loss_save_dir = 'plots/'
        # 确保基础目录存在
        os.makedirs(plot_loss_save_dir, exist_ok=True)
        # 构建完整的文件名
        plot_file_name = f"{base_save_name}_training_plot.png"
        # 构建完整的保存路径
        save_path = os.path.join(plot_loss_save_dir, plot_file_name)
        print(f"训练损失和指标绘图将被保存到: {save_path}")
        # 调用函数绘制并保存图表
        plot_training_history(history, additional_metrics=['dice_coefficient', 'iou', 'f1_score'], save_path=save_path)
def save_results_to_txt(config, test_results, file_path='training_results.txt'):
    """
    将模型配置、训练和测试结果保存到文本文件中。

    参数:
    - config: 包含模型配置和参数的字典。
    - test_results: 包含测试结果的字典。
    - file_path: 结果保存的文件路径。
    """
    with open(file_path, 'a') as file:
        file.write("Model Configuration and Results\n")
        file.write("-" * 40 + "\n")

        # 写入配置参数
        file.write("Configuration Parameters:\n")
        for key, value in config.items():
            file.write(f"{key}: {value}\n")

        # 写入测试结果
        file.write("Test Results:\n")
        for key, value in test_results.items():
            file.write(f"{key}: {value}\n")

        # 添加分隔符以分隔不同的训练结果
        file.write("=" * 40 + "\n\n")

    print(f"Results have been saved to: {file_path}")


if __name__ == "__main__":
    main()























