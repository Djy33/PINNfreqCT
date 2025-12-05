import numpy as np
import tensorflow as tf
from keras import backend as K

# --- 1. 数据增强函数 ---

def random_scaling(x, min_scale=0.9, max_scale=1.1, seed=None):
    """弱增强：对时频数据进行随机缩放。"""
    if seed is not None:
        np.random.seed(seed)
    # x shape: (T, F)
    scale_factor = np.random.uniform(min_scale, max_scale)
    return x * scale_factor


def random_permutation(x, segment_size=5, seed=None):
    """强增强：沿时间轴将数据按周期分段并随机置换，引入强时间扰动。"""
    if seed is not None:
        np.random.seed(seed)
    T = x.shape[0]
    num_segments = T // segment_size
    segments = []

    # 分割段
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size
        segments.append(x[start:end, :])

    if T % segment_size != 0:
        segments.append(x[num_segments * segment_size:, :])

    # 随机打乱
    shuffled_indices = np.random.permutation(len(segments))
    shuffled_segments = [segments[i] for i in shuffled_indices]

    # 重新组装
    return np.concatenate(shuffled_segments, axis=0)


# --- 2. 对比学习核心损失函数 (NT-Xent/InfoNCE Loss) ---

def contrastive_loss_base(hidden1, hidden2, temperature=0.2):
    """
    归一化温度缩放交叉熵损失 (NT-Xent)，用于上下文对比 (L_CC) 和时间对比 (L_TC)。
    """
    # 假设 hidden1, hidden2 是上下文特征 C 或序列特征 Z 的全局平均池化结果
    # shape: (B, D_c)

    # L2 归一化
    hidden1 = K.l2_normalize(hidden1, axis=1)
    hidden2 = K.l2_normalize(hidden2, axis=1)

    B = tf.shape(hidden1)[0]

    # 拼接所有表示: (2B, D_c)
    h_all = K.concatenate([hidden1, hidden2], axis=0)

    # 计算余弦相似度矩阵: (2B, 2B)，transpose_b=True会将第二个张量h_all转置
    sim_matrix = tf.matmul(h_all, h_all, transpose_b=True) / temperature

    # 掩盖对角线 (self-similarity)
    # mask_diag 生成一个对角线为 True 的矩阵（大小为2B×2B），代表了每个向量和自己之间的相似度
    # 第一个参数的位置用中间参数的值，其余用第三个参数的值
    mask_diag = tf.eye(2 * B, dtype=tf.bool)
    sim_matrix = tf.where(mask_diag, tf.constant(-1e9, dtype=tf.float32), sim_matrix)

    # 定义正样本掩码: (i, i+B) 和 (i+B, i)，true为正样本对，false为负样本对
    # 分块矩阵的右上角和左下角
    pos_mask = tf.concat([
        tf.concat([tf.zeros([B, B], dtype=tf.bool), tf.eye(B, dtype=tf.bool)], axis=1),
        tf.concat([tf.eye(B, dtype=tf.bool), tf.zeros([B, B], dtype=tf.bool)], axis=1)
    ], axis=0)

    # 提取正样本的 Logits (仅用于计算损失，不需要 tf.where)
    pos_logits = tf.where(pos_mask, sim_matrix, tf.constant(-1e9, dtype=tf.float32))

    # Log-Softmax 稳定计算 log(P_i/Sum(P_j))
    # 被减数部分：每个元素先取指数再每行求和最后取对数，keepdims=True 按行操作，保持行维不变，结果的形状为 2B×1
    # 结果为log_probs。
    # 对于每一个相似度，减去该行所有相似度的总和的对数，
    # 得到 归一化后的对数相似度。这样保证每行的对数相似度总和是 0，从而避免数值溢出。
    log_probs = sim_matrix - tf.reduce_logsumexp(sim_matrix, axis=1, keepdims=True)

    # 提取正样本的 Log Probability当样本对不是正样本时，填充 0
    # tf.zeros_like(log_probs)
    loss = -tf.reduce_sum(tf.where(pos_mask, log_probs, tf.zeros_like(log_probs)), axis=1) / 2.0

    return tf.reduce_mean(loss)


# --- 2. 时间对比损失函数 (体现时间性质) ---
def temporal_contrastive_loss(Z_w, Z_s, temperature=0.1, anchor_time=5, future_time=20):
    """
    时间对比损失 (L_TC)，使用序列特征 Z 进行跨视图的时间预测。

    Args:
        Z_w, Z_s (Tensor): 弱/强视图的序列特征, shape (B, T', D_z)
        anchor_time (int): 锚点（过去上下文）的索引 (t_anchor)
        future_time (int): 目标（未来预测）的索引 (t_future)
    """
    # 确保索引有效（防止超过序列长度）
    T_prime = tf.shape(Z_w)[1]
    anchor_idx = tf.minimum(anchor_time, T_prime - 2)
    future_idx = tf.minimum(future_time, T_prime - 1)

    # 1. L_TC^S: 用强视图的过去预测弱视图的未来
    # Anchor (C_past^S) = Z_s[:, t_anchor, :] (B, D_z)
    C_past_s = Z_s[:, anchor_idx, :]
    # Target (Z_future^W) = Z_w[:, t_future, :] (B, D_z)
    Z_future_w = Z_w[:, future_idx, :]
    L_tc_s = contrastive_loss_base(C_past_s, Z_future_w, temperature=temperature)

    # 2. L_TC^W: 用弱视图的过去预测强视图的未来
    C_past_w = Z_w[:, anchor_idx, :]
    Z_future_s = Z_s[:, future_idx, :]
    L_tc_w = contrastive_loss_base(C_past_w, Z_future_s, temperature=temperature)

    return L_tc_s + L_tc_w


# --- 3. 上下文对比损失函数 (全局不变性) ---
def contextual_contrastive_loss(C_w, C_s, temperature=0.2):
    """
    全局上下文对比损失 (L_CC)，作用于最终聚合特征 C。
    """
    return contrastive_loss_base(C_w, C_s, temperature=temperature)