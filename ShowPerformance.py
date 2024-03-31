import matplotlib.pyplot as plt


def plot_training_history(history, additional_metrics=[], save_path=None):
    """
    绘制训练和验证损失的变化曲线，以及其他指定的评估指标，并将图表保存到指定路径。

    参数:
    - history: Keras 训练过程返回的 History 对象。
    - additional_metrics: 列表，包含需要额外绘制的评估指标的名称。
    - save_path: 字符串，图表保存的路径和文件名。如果为 None，则不保存图表。
    """
    # 绘制训练和验证损失
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 如果指定了额外的评估指标，也绘制它们
    if additional_metrics:
        plt.subplot(1, 2, 2)
        for metric in additional_metrics:
            plt.plot(history.history[metric], label=f'Training {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'{metric.capitalize()} During Training')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()

    plt.tight_layout()

    # 如果指定了保存路径，保存图表
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存到：{save_path}")
    else:
        plt.show()



