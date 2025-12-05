from TFdata import *
from model import CNN, PINN, CNNfreq, PINNfreq, PINNfreqCT


# --- 辅助函数：提取预测结果 ---
def get_prediction_data(model, input_data, is_pinnct=False):
    """处理模型预测，返回非标准化的一维数组。"""
    if is_pinnct:
        # PINNfreqCT 返回 6 个输出，只取第一个 (Y^w_NN)
        pred_scaled = model(input_data)[0]
    else:
        # 其他模型返回 1 个输出
        pred_scaled = np.concatenate(model(input_data))[:, None]

    pred_unscaled = scaler_out.inverse_transform(pred_scaled)
    return np.concatenate(pred_unscaled)


# --- 辅助函数：绘制和保存对比图 ---
def create_comparison_plot(models_to_compare, filename, title):
    """
    绘制两张对比图（预测点 vs Loss曲线）并保存。
    models_to_compare: [(model, input_data, loss_train, loss_test, label, color, is_pinnct), ...]
    """
    # 1. 预测图 (s)
    s = figure(width=770, height=400, y_range=(out_min_rescaled - 20, out_max_rescaled + 20), title=title)
    # 绘制预测点
    for model, input_data, _, _, label, color, is_pinnct in models_to_compare:
        pred_data = get_prediction_data(model, input_data, is_pinnct)
        s.scatter(ix_test, pred_data, size=7, line_color=None, color=color, legend_label=label)
    # 真实 BP 曲线数据
    true_bp_line = np.concatenate(scaler_out.inverse_transform(all_out[:, 0][:, None]))
    # 绘制真实 BP 曲线
    s.line(range(len(df_demo_data)), true_bp_line, line_width=3,
           line_color='black', line_alpha=1, line_dash='dashed', legend_label='True BP')

    s.xaxis.axis_label = 'Beat time (s)'
    s.yaxis.axis_label = 'SBP (mmHg)'
    figure_settings(s)

    # 2. Loss 曲线图 (s2)
    s2 = figure(width=500, height=400, y_axis_type="log", y_range=(1e-2, 2), title=title)

    for _, _, loss_train, loss_test, label, color, _ in models_to_compare:
        # 训练损失 (实线)
        s2.line(np.linspace(0, 100, len(loss_train)), loss_train, line_width=3, alpha=0.8, color=color,
                legend_label=f'{label}-train')
        # 测试损失 (虚线)
        s2.line(np.linspace(0, 100, len(loss_test)), loss_test, line_width=3, line_dash='dashed',
                color=color, legend_label=f'{label}-test')

    s2.xaxis.axis_label = 'Training percent (%)'
    s2.yaxis.axis_label = 'mse norm.'
    figure_settings(s2)

    # 组合并保存
    result = row(s, s2)
    save(result, filename=f"fig/{filename}")
    print(f" Comparison saved to fig/{filename}")


if __name__ == '__main__':
    # model_dnn_cnn = CNN.cnn()
    # model_dnn_pinn = PINN.pinn()
    # model_dnn_cnnfreq = CNNfreq.cnn_freq()
    # model_dnn_pinnfreq = PINNfreq.pinn_freq()
    # model_dnn_pinnfreqct = PINNfreqCT.pinn_freq_ct()

##############################################################################
# 时域模型
    model_dnn_cnn = tf.keras.models.load_model("saved/model/CNN.h5")
    CNN.load_losses()
    model_dnn_pinn = tf.keras.models.load_model("saved/model/PINN.h5")
    PINN.load_losses()
# 时频域模型
    model_dnn_cnnfreq = tf.keras.models.load_model("saved/model/CNNfreq.h5")
    CNNfreq.load_losses()
    model_dnn_pinnfreq = tf.keras.models.load_model("saved/model/PINNfreq.h5")
    PINNfreq.load_losses()
    model_dnn_pinnfreqct = tf.keras.models.load_model("saved/model/PINNfreqCT.h5")
    PINNfreqCT.load_losses()
#PINN-TCC 的增强测试输入
    inp_comb_test_w_s = PINNfreqCT.prepare_augmented_tensors(PINNfreq.test_beats, test_feat1, test_feat2, test_feat3)
#使用索引 [0] 提取 Y^w_NN 预测值
    pinnct_pred_scaled = model_dnn_pinnfreqct(inp_comb_test_w_s)[0]
#时域模型测试输入
    inp_comb_test_t_domain = CNN.inp_comb_test
#其他时频域模型的测试输入
    inp_comb_test_ft_domain = PINNfreq.inp_comb_test
# ====================================================================
    # --- 目标对比 1: 时域 CNN vs 时域 PINN ---
    # 目的: 对比纯监督学习模型和加入物理约束的模型（时域）
    # ====================================================================
    comparison_1 = [
        (model_dnn_cnn, inp_comb_test_t_domain, CNN.loss_list_conv, CNN.test_loss_list_conv, 'CNN', palettes.Colorblind8[5], False),
        (model_dnn_pinn, inp_comb_test_t_domain, PINN.loss_list_pinn, PINN.test_loss_list_pinn, 'PINN', palettes.Colorblind8[3], False)
    ]
    create_comparison_plot(comparison_1, "CNN_vs_PINN.html", "物理模块性能对比")
    # ====================================================================
    # --- 目标对比 1+: 时频域 CNN vs 时频域 PINN ---
    # 目的: 对比纯监督学习模型和加入物理约束的模型（时频域）
    # ====================================================================
    comparison_11 = [
        (model_dnn_cnnfreq, inp_comb_test_ft_domain, CNNfreq.loss_list_conv, CNNfreq.test_loss_list_conv, 'CNNfreq', palettes.Colorblind8[5], False),
        (model_dnn_pinnfreq, inp_comb_test_ft_domain, PINNfreq.loss_list_pinn, PINNfreq.test_loss_list_pinn, 'PINNfreq', palettes.Colorblind8[3], False)
    ]
    create_comparison_plot(comparison_11, "CNNfreq_vs_PINNfreq.html", "物理模块性能对比")

    # ====================================================================
    # --- 目标对比 2: 时域 CNN vs 时频域 CNNfreq ---
    # 目的: 对比数据输入域的改变（时域 vs 时频域）
    # ====================================================================
    comparison_2 = [
        (model_dnn_cnn, inp_comb_test_t_domain, CNN.loss_list_conv, CNN.test_loss_list_conv, 'CNN', palettes.Colorblind8[5], False),
        (model_dnn_cnnfreq, inp_comb_test_ft_domain, CNNfreq.loss_list_conv, CNNfreq.test_loss_list_conv, 'CNNfreq', palettes.Colorblind8[3], False)
    ]
    create_comparison_plot(comparison_2, "CNN_vs_CNNfreq.html", "时频域性能对比")

    # ====================================================================
    # --- 目标对比 3: PINNfreq vs PINNfreqCT ---
    # 目的: 对比加入对比学习机制的 PINN 模型 (PINNfreqCT) 的效果
    # ====================================================================
    comparison_3 = [
        (model_dnn_pinnfreq, inp_comb_test_ft_domain, PINNfreq.loss_list_pinn, PINNfreq.test_loss_list_pinn, 'PINNfreq', palettes.Colorblind8[5], False),
        (model_dnn_pinnfreqct, inp_comb_test_w_s, PINNfreqCT.loss_list_hybrid, PINNfreqCT.test_loss_list_hybrid, 'PINNfreqCT', palettes.Colorblind8[3], True)
    ]
    create_comparison_plot(comparison_3, "PINNfreq_vs_PINNfreqCT.html", "对比学习性能对比")