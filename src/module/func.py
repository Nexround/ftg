import torch
import numpy as np
from scipy.optimize import minimize

import random
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch.nn as nn


class MinimalDotProductClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.num_labels = num_labels

        # Create the classifier layer
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize the classifier weights with the minimal dot product vectors
        classifier_weights = self._generate_minimal_dot_product_vectors(
            dim=hidden_size, num_labels=num_labels
        )

        self.classifier.weight.data = torch.tensor(
            classifier_weights, dtype=torch.float32
        )

        # Initialize the bias to zero
        self.classifier.bias.data.fill_(0.0)

        # Freeze the classifier parameters (set requires_grad=False)
        for param in self.classifier.parameters():
            param.requires_grad = False

    def _generate_minimal_dot_product_vectors(self, dim=128, num_labels=6):
        def loss_fn(flat_vectors):
            vectors = flat_vectors.reshape(num_labels, dim)
            dot_products = np.dot(vectors, vectors.T)
            loss = np.sum(dot_products) - np.trace(dot_products)
            return loss

        initial_vectors = np.random.randn(num_labels, dim)
        initial_vectors /= np.linalg.norm(initial_vectors, axis=1, keepdims=True)
        result = minimize(
            loss_fn,
            initial_vectors.flatten(),
            method="L-BFGS-B",
            options={"disp": True},
        )
        return result.x.reshape(num_labels, dim)

    def forward(self, x):
        # Check if x is 3D (batch_size, seq_len, hidden_size)
        if x.ndimension() == 3:
            x = x[:, 0, :]  # Take the first token/sequence element

        logits = self.classifier(x)
        return logits


def generate_minimal_dot_product_vectors(dim=128, num_labels=6):
    def loss_fn(flat_vectors):
        vectors = flat_vectors.reshape(num_labels, dim)
        dot_products = np.dot(vectors, vectors.T)
        loss = np.sum(dot_products) - np.trace(dot_products)
        return loss

    initial_vectors = np.random.randn(num_labels, dim)
    initial_vectors /= np.linalg.norm(initial_vectors, axis=1, keepdims=True)
    result = minimize(
        loss_fn, initial_vectors.flatten(), method="L-BFGS-B", options={"disp": True}
    )
    return result.x.reshape(num_labels, dim)


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


def scatter_plot(counter_obj, highlight_duplicates=None):
    x = [key[0] for key in counter_obj.keys()]
    y = [key[1] for key in counter_obj.keys()]
    sizes = [value * 0.001 for value in counter_obj.values()]  # 调整点大小以反映计数

    plt.scatter(
        x, y, s=sizes, alpha=0.6, color="skyblue", edgecolor="black", label="All Points"
    )

    # Highlight duplicate points if provided
    if highlight_duplicates:
        dup_x = [key[0] for key in highlight_duplicates]
        dup_y = [key[1] for key in highlight_duplicates]
        plt.scatter(dup_x, dup_y, s=50, alpha=0.9, color="red", label="Duplicates")

    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Scatter plot of (x, y) counts")
    plt.legend()
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


def generate_top_ig_triplets(per_layer_ig_values, per_knowledge_neuron_num):

    ig_triplets = []

    ig_tensor = torch.stack(per_layer_ig_values).to(dtype=torch.float16)
    ig_array = ig_tensor.numpy()
    ig_flattened = ig_array.flatten()
    top_K_indices = np.argpartition(ig_flattened, -per_knowledge_neuron_num)[
        -per_knowledge_neuron_num:
    ]
    row_indices, col_indices = np.unravel_index(top_K_indices, ig_array.shape)
    for layer_idx, neuron_idx  in zip(row_indices, col_indices):
        current_ig = ig_array[layer_idx, neuron_idx]
        ig_triplets.append([int(layer_idx), int(neuron_idx), float(current_ig)])

    return ig_triplets


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


def unfreeze_ffn_connections_with_hooks_optimized(model, trainable_neurons):
    hooks = []

    # 按层分组 trainable_neurons，减少 hook 数量

    layer_to_neurons = defaultdict(list)
    for layer, neuron_index in trainable_neurons:
        layer_to_neurons[layer].append(neuron_index)

    for layer, neuron_indices in layer_to_neurons.items():
        # 获取当前层的中间层和输出层
        intermediate_dense = model.bert.encoder.layer[layer].intermediate.dense
        output_dense = model.bert.encoder.layer[layer].output.dense

        # 确保权重和偏置的 requires_grad 为 True
        intermediate_dense.weight.requires_grad = True
        intermediate_dense.bias.requires_grad = True
        output_dense.weight.requires_grad = False  # 不允许调整输出层权重
        output_dense.bias.requires_grad = False

        # 注册单个钩子函数，针对输入权重的所有指定神经元
        def input_weight_hook(grad):
            mask = torch.zeros_like(grad)
            mask[neuron_indices, :] = 1  # 只允许指定神经元的梯度
            return grad * mask

        hooks.append(intermediate_dense.weight.register_hook(input_weight_hook))

        # 注册单个钩子函数，针对输出权重的所有指定神经元
        # def output_weight_hook(grad):
        #     mask = torch.zeros_like(grad)
        #     mask[:, neuron_indices] = 1  # 只允许指定神经元的梯度
        #     return grad * mask

        # hooks.append(output_dense.weight.register_hook(output_weight_hook))

        # 注册单个钩子函数，针对输入偏置的所有指定神经元
        def input_bias_hook(grad):
            mask = torch.zeros_like(grad)
            mask[neuron_indices] = 1  # 只允许指定神经元的梯度
            return grad * mask

        hooks.append(intermediate_dense.bias.register_hook(input_bias_hook))

        # 输出层偏置不需要区分神经元，保持全梯度
        # def output_bias_hook(grad):
        #     mask = torch.zeros_like(grad)
        #     return grad * mask  # 不允许调整输出层bias

        # hooks.append(output_dense.bias.register_hook(output_bias_hook))

    return hooks  # 返回 hooks 以便后续管理和清理


def unfreeze_ffn_qwen(model, trainable_neurons):
    hooks = []

    # 按层分组 trainable_neurons，减少 hook 数量

    layer_to_neurons = defaultdict(list)
    for layer, neuron_index in trainable_neurons:
        layer_to_neurons[layer].append(neuron_index)

    for layer, neuron_indices in layer_to_neurons.items():
        # 获取当前层的中间层和输出层
        # intermediate_dense = model.bert.encoder.layer[layer].mlp.gate_proj
        output_dense = model.model.layers[layer].mlp.down_proj
        output_dense.requires_grad = True
        # 确保权重和偏置的 requires_grad 为 True
        # intermediate_dense.weight.requires_grad = True
        # intermediate_dense.bias.requires_grad = True
        output_dense.weight.requires_grad = True

        # 注册单个钩子函数，针对输入权重的所有指定神经元
        # def input_weight_hook(grad):
        #     mask = torch.zeros_like(grad)
        #     mask[neuron_indices, :] = 1  # 只允许指定神经元的梯度
        #     return grad * mask

        # hooks.append(intermediate_dense.weight.register_hook(input_weight_hook))

        # 注册单个钩子函数，针对输出权重的所有指定神经元
        def output_weight_hook(grad):
            mask = torch.zeros_like(grad)
            mask[:, neuron_indices] = 1  # 只允许指定神经元的梯度
            return grad * mask

        hooks.append(output_dense.weight.register_hook(output_weight_hook))

        # 注册单个钩子函数，针对输入偏置的所有指定神经元
        # def input_bias_hook(grad):
        #     mask = torch.zeros_like(grad)
        #     mask[neuron_indices] = 1  # 只允许指定神经元的梯度
        #     return grad * mask

        # hooks.append(intermediate_dense.bias.register_hook(input_bias_hook))

        # 输出层偏置不需要区分神经元，保持全梯度
        # def output_bias_hook(grad):
        #     mask = torch.zeros_like(grad)
        #     return grad * mask  # 不允许调整输出层bias

        # hooks.append(output_dense.bias.register_hook(output_bias_hook))

    return hooks  # 返回 hooks 以便后续管理和清理


def compute_metrics(pred, method="weighted"):
    """
    计算模型的评价指标，包括准确率、精确率、召回率和F1分数。

    Args:
        pred: 一个包含两个字段的对象，分别是`predictions`和`label_ids`。
              `predictions`是模型的预测结果，`label_ids`是真实标签。

    Returns:
        Dict[str, float]: 各种评价指标的字典。
    """
    # 获取预测结果和真实标签
    predictions = pred.predictions.argmax(axis=-1)  # 对每个样本选择概率最高的类别
    label_ids = pred.label_ids

    # 计算评价指标
    accuracy = accuracy_score(label_ids, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_ids, predictions, average=method  # 加权平均，适合多分类
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
