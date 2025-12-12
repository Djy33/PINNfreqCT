import numpy as np
import tensorflow as tf
import os
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import row
from bokeh import palettes

# 1. 导入项目数据和配置
# 必须导入 df_demo_data, ix_test, all_out 等全局变量以还原坐标轴
from TFdata import *
from model import PINNfreq, PINNfreqCT
from miscFun import save_svg_and_pdf, figure_settings

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)


# --- 核心辅助函数 ---

def prepare_deterministic_tensors(beats, feat1, feat2, feat3):
    """
    测试专用：不进行随机 Scaling，保证 noise=0 时结果稳定且唯一。
    直接将输入作为 Weak View。
    """
    if beats.dtype == object:
        beats = np.vstack(beats).astype(np.float32)
    else:
        beats = beats.astype(np.float32)

    # 推理时，Weak View 和 Strong View 都用原始(或加噪)数据填充
    inp_beat_w = tf.convert_to_tensor(beats, dtype=tf.float32)
    inp_beat_s = tf.convert_to_tensor(beats, dtype=tf.float32)
    inp_feat1 = tf.convert_to_tensor(feat1, dtype=tf.float32)
    inp_feat2 = tf.convert_to_tensor(feat2, dtype=tf.float32)
    inp_feat3 = tf.convert_to_tensor(feat3, dtype=tf.float32)

    return [inp_beat_w, inp_beat_s, inp_feat1, inp_feat2, inp_feat3]


def ensure_numeric(data):
    """数据清洗：Object -> Float32"""
    try:
        if data.dtype == object:
            return np.vstack(data).astype(np.float32)
        return data.astype(np.float32)
    except Exception:
        return data


def add_noise(data, noise_level):
    """加噪函数"""
    if noise_level == 0: return data
    noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
    return data + noise



# --- 绘图函数：完全复刻 main.py ---

def create_comparison_plot_full(pred_base, pred_pinn, pred_ct, noise_level):
    """
    绘制对比图并保存为 SVG/PDF
    """
    title = f"Robustness Test (Noise Level = {noise_level})"
    filename = f"noise_comparison_{noise_level}"  # 不带后缀

    # 1. 真实 BP 曲线
    true_bp_line = np.concatenate(scaler_out.inverse_transform(all_out[:, 0][:, None]))

    # 2. 创建画布
    s = figure(width=900, height=450,
               y_range=(out_min_rescaled - 20, out_max_rescaled + 20),
               title=title)
    s.output_backend = "svg"

    # 3. 绘制预测点
    # base (Time Domain)
    s.scatter(ix_test, pred_base, size=7, line_color=None,
              color=palettes.Colorblind8[5], legend_label='base')
    # 3. 绘制预测点
    # PINNfreq
    # s.scatter(ix_test, pred_pinn, size=7, line_color=None,
    #           color=palettes.Colorblind8[4], legend_label='PINNfreq')

    # PINNfreqCT (Contrastive)
    s.scatter(ix_test, pred_ct, size=7, line_color=None,
              color=palettes.Colorblind8[3], legend_label='PINNfreqCT')

    # 4. 绘制真实线
    s.line(range(len(df_demo_data)), true_bp_line, line_width=3,
           line_color='black', line_alpha=1, line_dash='dashed', legend_label='True BP')

    s.xaxis.axis_label = 'Beat time (s)'
    s.yaxis.axis_label = 'SBP (mmHg)'

    # 应用样式
    figure_settings(s)


    # 5. 使用 save_svg_and_pdf 保存
    # 注意：save_svg_and_pdf 内部已经包含了 .svg 和 .pdf 的转换逻辑
    save_svg_and_pdf(s, filename)



def save_metrics_plot(noise_levels, results_rmse, results_corr):
    """绘制 RMSE/Corr 趋势图"""
    # RMSE
    p1 = figure(width=500, height=400, title="RMSE vs Noise (Lower is Better)",
                x_axis_label='Noise Level', y_axis_label='RMSE (mmHg)')
    p1.output_backend = "svg"
    p1.line(noise_levels, results_rmse['base'], width=3, color=palettes.Colorblind8[5], legend_label='base')
    p1.circle(noise_levels, results_rmse['base'], size=8, color=palettes.Colorblind8[5], legend_label='base')
    p1.line(noise_levels, results_rmse['PINNfreq'], width=3, color=palettes.Colorblind8[4], legend_label='PINN')
    p1.circle(noise_levels, results_rmse['PINNfreq'], size=8, color=palettes.Colorblind8[4], legend_label='PINN')
    p1.line(noise_levels, results_rmse['PINNfreqCT'], width=3, color=palettes.Colorblind8[3], legend_label='PINNfreqCT')
    p1.square(noise_levels, results_rmse['PINNfreqCT'], size=8, color=palettes.Colorblind8[3],
              legend_label='PINNfreqCT')

    # Correlation
    p2 = figure(width=500, height=400, title="Corr vs Noise (Higher is Better)",
                x_axis_label='Noise Level', y_axis_label='Correlation')
    p2.output_backend = "svg"
    p2.line(noise_levels, results_corr['base'], width=3, color=palettes.Colorblind8[4], legend_label='base')
    p2.circle(noise_levels, results_corr['base'], size=8, color=palettes.Colorblind8[4], legend_label='base')
    p2.line(noise_levels, results_corr['PINNfreq'], width=3, color=palettes.Colorblind8[5], legend_label='PINN')
    p2.circle(noise_levels, results_corr['PINNfreq'], size=8, color=palettes.Colorblind8[5], legend_label='PINN')
    p2.line(noise_levels, results_corr['PINNfreqCT'], width=3, color=palettes.Colorblind8[3], legend_label='PINNfreqCT')
    p2.square(noise_levels, results_corr['PINNfreqCT'], size=8, color=palettes.Colorblind8[3],
              legend_label='PINNfreqCT')
    p2.legend.location = "bottom_left"

    # 组合并通过 SVG 保存
    combined_layout = row(p1, p2)
    save_svg_and_pdf(combined_layout, "noise_metrics_summary")



# --- 主流程 ---

def run_noise_experiment():
    print("=== 1. 准备数据 ===")
    # 清洗数据
    raw_test_beats = ensure_numeric(PINNfreq.test_beats)
    feat1 = ensure_numeric(PINNfreq.test_feat1)
    feat2 = ensure_numeric(PINNfreq.test_feat2)
    feat3 = ensure_numeric(PINNfreq.test_feat3)
    # 目标值 (用于计算指标)
    test_out_scaled = ensure_numeric(PINNfreq.test_out)

    print("=== 2. 加载模型 ===")
    model_base = tf.keras.models.load_model("../saved/model/PINN.h5", compile=False)
    model_pinn = tf.keras.models.load_model("../saved/model/PINNfreq.h5")
    model_ct = tf.keras.models.load_model("../saved/model/PINNfreqCT.h5")

    # 定义噪声等级
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    results_rmse = {'base': [],'PINNfreq': [], 'PINNfreqCT': []}
    results_corr = {'base': [],'PINNfreq': [], 'PINNfreqCT': []}

    print("\n=== 3. 开始测试 ===")
    for level in noise_levels:
        print(f"Testing Noise Level: {level}")

        # 加噪 (仅对 beats)
        noisy_beats = add_noise(raw_test_beats, level)

        # A. 预测 - PINN (Time Domain)
        # PINN 时域模型通常需要 (N, T, 1) 的输入
        pred_scaled_base = model_base.predict([noisy_beats[:,:,0], feat1, feat2, feat3], verbose=0)
        if isinstance(pred_scaled_base, list): pred_scaled_base = pred_scaled_base[0]
        # 反标准化
        pred_base_real = scaler_out.inverse_transform(pred_scaled_base.reshape(-1, 1)).flatten()

        # B. 预测 - PINNfreq
        pred_scaled_pinn = model_pinn.predict([noisy_beats, feat1, feat2, feat3], verbose=0)
        if isinstance(pred_scaled_pinn, list): pred_scaled_pinn = pred_scaled_pinn[0]
        # 反标准化
        pred_pinn_real = scaler_out.inverse_transform(pred_scaled_pinn.reshape(-1, 1)).flatten()

        # C. 预测 - PINNfreqCT (使用 prepare_deterministic_tensors)
        inp_ct = prepare_deterministic_tensors(noisy_beats, feat1, feat2, feat3)
        pred_scaled_ct = model_ct.predict(inp_ct, verbose=0)[0]
        # 反标准化
        pred_ct_real = scaler_out.inverse_transform(pred_scaled_ct.reshape(-1, 1)).flatten()

        # D. 计算指标
        # 真实值反标准化
        true_real = scaler_out.inverse_transform(test_out_scaled.reshape(-1, 1)).flatten()

        rmse_base = np.sqrt(np.mean((true_real - pred_base_real) ** 2))
        corr_base = np.corrcoef(true_real, pred_base_real)[0, 1]

        rmse_pinn = np.sqrt(np.mean((true_real - pred_pinn_real) ** 2))
        corr_pinn = np.corrcoef(true_real, pred_pinn_real)[0, 1]

        rmse_ct = np.sqrt(np.mean((true_real - pred_ct_real) ** 2))
        corr_ct = np.corrcoef(true_real, pred_ct_real)[0, 1]

        results_rmse['base'].append(rmse_base)
        results_corr['base'].append(corr_base)
        results_rmse['PINNfreq'].append(rmse_pinn)
        results_corr['PINNfreq'].append(corr_pinn)
        results_rmse['PINNfreqCT'].append(rmse_ct)
        results_corr['PINNfreqCT'].append(corr_ct)

        print(f"  RMSE -> base: {rmse_base:.2f} | PINN: {rmse_pinn:.2f} | CT: {rmse_ct:.2f}")

        # E. 画图 (使用全集 scatter 风格)
        # 我们在 0.0 (基准) 和 0.2 (典型噪声) 时保存对比图
        if level in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            create_comparison_plot_full(pred_base_real, pred_pinn_real, pred_ct_real, level)

    # 4. 保存趋势图
    save_metrics_plot(noise_levels, results_rmse, results_corr)


if __name__ == '__main__':
    run_noise_experiment()