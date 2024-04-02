




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
    "augmentation": True,

    "mode": "ThreeBand",
    "optimizer_name": "Adadelta",
    "learning_rate": 1.0,
    "rho": 0.95,
    "epsilon": 1e-8,
    "loss_name": "dice_loss",
    "metrics_names": ["dice_coefficient", "iou", "f1_score"]
}




