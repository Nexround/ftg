import torch
import numpy as np
import random


def scaled_input(emb, batch_size, num_batch):
    """
    生成一组按比例缩放的输入数据。

    参数:
        emb (torch.Tensor): 输入张量，形状为 (1, ffn_size)，表示一个基准向量。
        batch_size (int): 每个批次中的样本数量。
        num_batch (int): 批次的总数量。

    返回:
        res (torch.Tensor): 形状为 (batch_size * num_batch, ffn_size) 的张量，包含生成的缩放输入。
        step (torch.Tensor): 每次增量的大小，形状为 (1, ffn_size)，表示每一步的变化量。
    """

    # baseline: 初始化为与 emb 形状相同的零张量。用于生成与 emb 相比按比例增大的输入。
    baseline = torch.zeros_like(emb)  # (1, ffn_size)

    # num_points: 总的输入样本数，等于 batch_size 乘以 num_batch
    num_points = batch_size * num_batch

    # step: 计算每一步的增量，即 emb 和 baseline 之间的差距除以总的样本数。
    # 这是为了确保从 baseline 到 emb 的变化是均匀的，按比例分配给每个样本。
    step = (emb - baseline) / num_points  # (1, ffn_size)

    # res: 使用 baseline 和 step 生成 scaled input 的列表。
    # 通过 torch.add 和 step * i 逐步构建每个样本的值，并将这些值沿着第一个维度 (batch_size * num_batch) 拼接。
    res = torch.cat(
        [torch.add(baseline, step * i) for i in range(num_points)], dim=0
    )  # (num_points, ffn_size)

    # 返回生成的输入数据和每一步的增量。
    return res, step[0]


import matplotlib.pyplot as plt


def scatter_plot(counter_obj):
    x = [key[0] for key in counter_obj.keys()]
    y = [key[1] for key in counter_obj.keys()]
    sizes = [value * 10 for value in counter_obj.values()]  # 调整点大小以反映计数

    plt.scatter(x, y, s=sizes, alpha=0.6, color="skyblue", edgecolor="black")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Scatter plot of (x, y) counts")
    plt.show()


def heatmap_plot(counter_obj):
    max_x = max(key[0] for key in counter_obj.keys()) + 1
    max_y = max(key[1] for key in counter_obj.keys()) + 1
    heatmap = np.zeros((max_x, max_y))

    for (x, y), count in counter_obj.items():
        heatmap[x, y] = count

    plt.imshow(heatmap, cmap="Blues", origin="lower")
    plt.colorbar(label="Counts")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Heatmap of (x, y) counts")
    plt.show()


def plot_points(points, color="blue", marker="o", title="Scatter Plot of Points"):
    """
    绘制包含 (x, y) 对的集合。

    参数:
        points (set): 包含 (x, y) 对的集合。
        color (str): 点的颜色 (默认为 'blue')。
        marker (str): 点的标记样式 (默认为 'o')。
        title (str): 图形标题 (默认为 'Scatter Plot of Points')。
    """
    if not points:
        print("集合为空，无法绘制。")
        return

    # 分离集合中的 x 和 y 坐标
    x_values, y_values = zip(*points)

    # 绘制散点图
    plt.scatter(x_values, y_values, color=color, marker=marker, label="Points")

    # 添加图形元素
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.show()


def convert_to_triplet_ig_top(ig_list, RETENTION_THRESHOLD=99):
    # ig_list 12个transformer block，每个block 3072个ffn中间层激活权重
    # i：Transformer 层的索引（0 到 11，共 12 层）
    # j：每个层中 FFN 中间层激活权重的索引（0 到 3071，共 3072 个中间层）
    # ig[i][j]：集成梯度值，即该位置的权重。
    ig_triplet = []
    ig = np.array(ig_list)  # 12, 3072
    # 计算前5%的阈值
    threshold = np.percentile(ig, RETENTION_THRESHOLD)

    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= threshold:
                ig_triplet.append([i, j, ig[i][j]])

    return ig_triplet


def convert_to_triplet_ig(ig_list):
    # ig_list 12个transformer block，每个block 3072个ffn中间层激活权重
    # i：Transformer 层的索引（0 到 11，共 12 层）
    # j：每个层中 FFN 中间层激活权重的索引（0 到 3071，共 3072 个中间层）
    # ig[i][j]：集成梯度值，即该位置的权重。
    ig_triplet = []
    ig = np.array(ig_list)  # 12, 3072
    max_ig = ig.max()
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


def parse_comma_separated(input_string):
    # 将逗号分隔的字符串转换为元组
    return tuple(input_string.split(","))


def extract_random_samples(dataset, num_samples):
    """
    从给定的数据集中随机提取指定数量的样本，忽略长度为 1 的文本。

    参数：
    - dataset: 数据集，需包含类似于 Hugging Face 数据集的结构，带有键 "train" 和 "text"。
    - num_samples: 需要提取的样本数量。

    返回：
    - 一个包含随机样本的列表。
    """
    # 获取训练数据中的文本，忽略空文本和长度为 1 的文本
    text_data = [
        text
        for text in dataset["train"]["text"]
        if text.strip() != "" and len(text.strip()) > 1
    ]

    # 检查请求的样本数是否超过数据集大小
    if num_samples > len(text_data):
        raise ValueError(
            f"请求的样本数量 ({num_samples}) 超过可用文本数量 ({len(text_data)})。"
        )

    # 随机选择指定数量的样本
    random_samples = random.sample(text_data, num_samples)

    return random_samples


def remove_empty_dimensions(lst):
    # 如果列表是空的，返回空列表
    if lst == []:
        return []

    # 如果列表只有一个元素，且这个元素本身是列表
    # 就递归地去除其中的空维度
    if isinstance(lst, list):
        # 对每个元素调用递归函数
        lst = [remove_empty_dimensions(item) for item in lst]
        # 如果这个列表包含的元素为空列表，则去除这个空列表
        while len(lst) == 1 and isinstance(lst[0], list) and lst[0] == []:
            lst = lst[0]
    return lst
