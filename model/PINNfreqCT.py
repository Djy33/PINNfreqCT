from CTNN import *
from CT import *
from TFdata import *
import pickle

# 在训练完成后保存损失列表
def save_losses(loss_list, test_loss_list, save_path="./saved/loss/losses_PINNfreqCT.pkl"):
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存数据
    with open(save_path, 'wb') as f:
        pickle.dump({
            'loss_list_hybrid': loss_list,
            'test_loss_list_hybrid': test_loss_list
        }, f)
    print(f"损失数据已保存至：{save_path}")

# 加载损失列表
def load_losses(load_path="./saved/loss/losses_PINNfreqCT.pkl"):
    global loss_list_hybrid, test_loss_list_hybrid
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    loss_list_hybrid = data['loss_list_hybrid']
    test_loss_list_hybrid = data['test_loss_list_hybrid']


# 准备强/弱增强的数据输入
def prepare_augmented_tensors(beats, feat1, feat2, feat3):
    # 1. 内存中 numpy 增强
    beats_w_list = []
    beats_s_list = []
    for beat in beats:
        # 生成弱增强和强增强
        beats_w_list.append(random_scaling(beat, min_scale=0.9, max_scale=1.05))
        beats_s_list.append(random_permutation(random_scaling(beat, min_scale=0.8, max_scale=1.2), segment_size=feat1.shape[0]))

    beats_w = np.stack(beats_w_list, axis=0)
    beats_s = np.stack(beats_s_list, axis=0)

    # 2. 转换为 TF Tensors
    inp_beat_w = tf.convert_to_tensor(beats_w, dtype=tf.float32)
    inp_beat_s = tf.convert_to_tensor(beats_s, dtype=tf.float32)
    inp_feat1 = tf.convert_to_tensor(feat1, dtype=tf.float32)
    inp_feat2 = tf.convert_to_tensor(feat2, dtype=tf.float32)
    inp_feat3 = tf.convert_to_tensor(feat3, dtype=tf.float32)

    return [inp_beat_w, inp_beat_s, inp_feat1, inp_feat2, inp_feat3]

# 损失和指标列表
loss_list_hybrid = []
test_loss_list_hybrid = []

def pinn_freq_ct():

    T = train_beats.shape[1]
    F = train_beats.shape[2]
    N_EXT = 64  # CNN 提取特征数量（可调）
    D_C = 60  # Contextual Feature C 维度（可调）

# --- 初始化 PINN-TCC 混合模型 ---
    full_model, pred_head, encoder = model_PINN_TCC(T, F, N_EXT=N_EXT, D_C=D_C)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)  # 调整学习率
# 损失权重
    ALPHA = 1.0  # L_conventional 权重
    BETA = 1.0  # L_physics 权重
    GAMMA = 1.0  # L_TC 权重
    DELTA = 0.7  # L_CC 权重


    epochs = 5000
    for epoch in range(epochs):
    # --- 1. 准备增强输入 ---
    # 对小批量训练集 (L) 进行增强
        inp_comb_train_w_s = prepare_augmented_tensors(train_beats, train_feat1, train_feat2, train_feat3)

    # 对全集 (ALL) 进行增强 (用于 L_physics 的前向传播)
        inp_comb_all_aug = prepare_augmented_tensors(all_beats, all_feat1, all_feat2, all_feat3)

        with tf.GradientTape() as tape:
        # 2. 前向传播 (Labeled Batch L)
        # Outputs: [y_nn_w, y_nn_s, Z_w, Z_s, C_w, C_s]
            y_nn_w, y_nn_s, Z_w, Z_s, C_w, C_s = full_model(inp_comb_train_w_s, training=True)

        # --- A. L_conventional (监督损失) ---
        # 默认使用弱视图预测，与真值 train_out 对比
            L_conv = K.mean(K.square(y_nn_w - train_out))

        # --- B. L_TC (时间对比损失) ---
        # Z_w/Z_s 的形状为 [Batch, T', D_z]
            L_tc = temporal_contrastive_loss(Z_w, Z_s, temperature=0.5)

        # --- C. L_CC (上下文对比损失) ---
        # 简化：对全局特征 C 进行非监督对比 (L_CC)，因为没有类别伪标签,BP是连续值，将只在y_{true}完全相同时才认为它们是正样本，过于严格
        # 真正的 L_CC 需要知道类别，并调整 contrastive_loss 中的正负样本对
            L_cc = contextual_contrastive_loss(C_w, C_s, temperature=0.2)

        # --- D. L_physics (物理损失) ---
        # 沿用原 PINN 逻辑，使用全集数据 inp_comb_all_aug (这里是强视图 beat_s_all)
            with tf.GradientTape(persistent=True) as physics_tape:
            # physics_tape 监视 U (生理特征)
                physics_tape.watch([inp_comb_all_aug[2], inp_comb_all_aug[3], inp_comb_all_aug[4]])

            # 使用 Shared Encoder + Prediction Head 预测全集 Y_NN
            # 仅使用强视图 beat_s_all 及其对应的特征
                inp_beat_s_all = inp_comb_all_aug[1]
                feat1_inp_all = inp_comb_all_aug[2]
                feat2_inp_all = inp_comb_all_aug[3]
                feat3_inp_all = inp_comb_all_aug[4]

                _, feat_ext_all = encoder(inp_beat_s_all)
                feat_comb_all = Concatenate()([feat1_inp_all, feat2_inp_all, feat3_inp_all, feat_ext_all])
                _, y_nn_all = pred_head(feat_comb_all)

        # 计算 dY_NN / dU
            dx_f1 = physics_tape.gradient(y_nn_all, feat1_inp_all)
            dx_f2 = physics_tape.gradient(y_nn_all, feat2_inp_all)
            dx_f3 = physics_tape.gradient(y_nn_all, feat3_inp_all)
        #释放内存
            del physics_tape

        # Taylor 近似残差 (h) 计算 (沿用原笔记本逻辑)
            pred_physics = (y_nn_all[:-1, 0]
                        + Multiply()([dx_f1[:-1, 0], feat1_inp_all[1:, 0] - feat1_inp_all[:-1, 0]])
                        + Multiply()([dx_f2[:-1, 0], feat2_inp_all[1:, 0] - feat2_inp_all[:-1, 0]])
                        + Multiply()([dx_f3[:-1, 0], feat3_inp_all[1:, 0] - feat3_inp_all[:-1, 0]])
                        )
            physics_residual = pred_physics - y_nn_all[1:, 0]
        # 仅在非忽略索引 ix_all[:-1] 上计算损失 (这里为简化，直接在所有连续点上计算)
            L_physics = K.mean(K.square(physics_residual))

        # --- E. L_total (总损失) ---
            L_total = (ALPHA * L_conv + BETA * L_physics +
                   GAMMA * L_tc + DELTA * L_cc)

    # 4. 应用梯度
        grads = tape.gradient(L_total, full_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, full_model.trainable_weights))
        loss_list_hybrid.append(float(L_conv))
        loss_final=L_total

        inp_comb_test_w_s = prepare_augmented_tensors(test_beats, test_feat1, test_feat2, test_feat3)
        y_nn_w, y_nn_s, Z_w, Z_s, C_w, C_s = full_model(inp_comb_test_w_s)
    #训练过程也是用的弱视图（时序数据尽可能的保持原状）
        test_loss_ini = y_nn_w - test_out
        test_loss = K.mean(K.square(test_loss_ini))
        test_loss_list_hybrid.append(float(test_loss))

    # break condition (使用 L_conv)
        if (loss_final <= 0.01) | (epoch == epochs - 1):
            # 打印损失
            print(
                "Hybrid PINN-TCC Completed. Epoch %d/%d -- L_total: %.4f (L_conv: %.4f, L_physics: %.4f, L_tc: %.4f, L_cc: %.4f)" %
                (epoch, epochs, float(L_total), float(L_conv), float(L_physics), float(L_tc), float(L_cc)))
        # 退出循环
            break

    inp_comb_test_w_s = prepare_augmented_tensors(test_beats, test_feat1, test_feat2, test_feat3)
    y_nn_w, y_nn_s, Z_w, Z_s, C_w, C_s = full_model(inp_comb_test_w_s)
    corr_pinn = np.corrcoef(np.concatenate(test_out)[:], np.concatenate(y_nn_w)[:])[0][1]
    rmse_pinn = np.sqrt(np.mean(np.square(
        np.concatenate(scaler_out.inverse_transform(np.concatenate(test_out)[:][:, None])) -
        np.concatenate(scaler_out.inverse_transform(np.concatenate(y_nn_w)[:][:, None])))))
    print('#### PINNfreqCT Performance ####')
    print('Corr: %.2f,  RMSE: %.1f' % (corr_pinn, rmse_pinn))

    model_save_path = "./saved/model/PINNfreqCT.h5"  # HDF5格式
    full_model.save(model_save_path)
    print(f"模型已保存至：{model_save_path}")
    # 训练结束后保存损失
    save_losses(loss_list_hybrid, test_loss_list_hybrid)

    return full_model