# 修改后的模型架构函数，定义共享的 Beat 编码器
from tensorflow.keras.layers import Concatenate, Dense, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from CT import *



# 输出[用于时间对比的时序特征，用于上下文对比和PINN的特征]
def create_shared_encoder(T, F, N_EXT=64):
    """创建共享的 Beat 时频数据编码器"""
    inp_beat = Input(shape=(T, F))

    # Conv1D Layers
    cnn1_1 = Conv1D(32, 5, activation='relu')(inp_beat)
    cnn1_2 = Conv1D(64, 3, activation='relu')(cnn1_1)
    mp_cnn1 = MaxPooling1D(pool_size=3, strides=1)(cnn1_2)

    # --- 输出 1: 序列特征 Z for L_TC (保留时间维度) ---
    Z = mp_cnn1

    # 后续接 Flatten 和 Dense
    fl_cnn1 = Flatten()(Z)
    feat_ext = Dense(N_EXT, activation='relu', name='feat_ext')(fl_cnn1)

    # 输出: [Z, feat_ext]
    encoder = Model(inputs=inp_beat, outputs=[Z, feat_ext], name='Beat_Encoder')
    return encoder


def model_PINN_TCC(T, F, N_EXT=64, D_C=60):
    # 1. 创建共享编码器
    encoder = create_shared_encoder(T, F, N_EXT=N_EXT)

    # 2. 定义输入: W/S 视图的 Beat 输入 + 3个不变的生理特征输入
    inp_beat_w = Input(shape=(T, F), name='beat_w')
    inp_beat_s = Input(shape=(T, F), name='beat_s')
    inp_feat1 = Input(shape=(1,), name='feat1')
    inp_feat2 = Input(shape=(1,), name='feat2')
    inp_feat3 = Input(shape=(1,), name='feat3')

    # 3. 定义可共享的预测头 (Contextual Feature C -> Y_NN)
    # 关键点：这一部分权重必须在 W/S 视图间共享

    # 共享的预测头：从拼接特征到预测输出
    inp_feat_comb = Input(shape=(N_EXT + 3,), name='feat_comb_input')

    # --- 输出 2: 上下文特征 C for L_SCC/L_CC ---
    C = Dense(D_C, activation='relu', name='Context_Feature')(inp_feat_comb)
    # --- 输出 3: 最终预测 Y_NN ---
    y_nn = Dense(1, name='Prediction')(C)

    prediction_head = Model(inputs=inp_feat_comb, outputs=[C, y_nn], name='Prediction_Head')

    # 4. 组装完整模型（Weak View）
    Z_w, feat_ext_w = encoder(inp_beat_w)
    feat_comb_w = Concatenate(name='concat_w')([inp_feat1, inp_feat2, inp_feat3, feat_ext_w])
    C_w, y_nn_w = prediction_head(feat_comb_w)

    # 5. 组装完整模型（Strong View）
    Z_s, feat_ext_s = encoder(inp_beat_s)
    feat_comb_s = Concatenate(name='concat_s')([inp_feat1, inp_feat2, inp_feat3, feat_ext_s])
    C_s, y_nn_s = prediction_head(feat_comb_s)

    # 6. 定义最终的 TCC-PINN 模型
    full_model = Model(
        inputs=[inp_beat_w, inp_beat_s, inp_feat1, inp_feat2, inp_feat3],
        outputs=[y_nn_w, y_nn_s, Z_w, Z_s, C_w, C_s],  # 输出所有计算损失所需项
        name='PINN_TCC_Hybrid_Model'
    )
    return full_model, prediction_head, encoder

