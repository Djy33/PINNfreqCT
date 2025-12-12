from miscFun import figure_settings,save_svg_and_pdf
from funetuning import *


# ==========================
# 评估与绘图
# ==========================
def evaluate(model, test_in, true_bp, type):
    if type == 'PINNfreqCT':
        inputs = [test_in[0], test_in[0], test_in[1], test_in[2], test_in[3]]
        pred = model.predict(inputs, verbose=0)[0]
    elif type == 'PINNfreq':
        pred = model.predict(test_in, verbose=0)
        if isinstance(pred, list): pred = pred[0]
    else:
        inputs = [tf.expand_dims(test_in[0][:,:,0], axis=-1),test_in[1], test_in[2], test_in[3]]
        pred = model.predict(inputs, verbose=0)
        if isinstance(pred, list): pred = pred[0]

    pred_val = pipeline.scaler_out.inverse_transform(pred).flatten()
    rmse = np.sqrt(np.mean((true_bp - pred_val) ** 2))
    return pred_val, rmse

def make_plot(true, p0, p1, p2, title, r0, r1, r2):
    p = figure(width=900, height=400, title=f"{title} | RMSE: PINN={r1:.1f}, CT={r2:.1f}",
                   x_axis_label='Heart Beat Index', y_axis_label='SBP (mmHg)')
    p.output_backend = "svg"
    p.scatter(range(len(p0)), p0, color=palettes.Colorblind8[5], legend_label='PINN', size=6)
    # p.scatter(range(len(p1)), p1, color=palettes.Colorblind8[4], legend_label='PINNfreq', size=6)
    p.scatter(range(len(p2)), p2, color=palettes.Colorblind8[3], legend_label='PINNfreqCT', size=6)
    p.line(range(len(true)), true, color='black', line_dash='dashed', legend_label='True BP', line_width=2, line_alpha=1)
    figure_settings(p)
    return p

# ==========================
# 生成综合评估图表：Scatter, Bland-Altman, Time-Series
# ==========================
def plot_comprehensive_analysis(true_bp, pred_bp, title_prefix, color):

    # 计算指标
    rmse = np.sqrt(np.mean((true_bp - pred_bp) ** 2))
    corr = np.corrcoef(true_bp, pred_bp)[0, 1]
    bias = np.mean(pred_bp - true_bp)
    std_diff = np.std(pred_bp - true_bp)

        # --- 图 1: 散点回归图 (Regression Plot) ---
        # 作用：展示预测值与真实值的相关性。越接近对角线越好。
    p_scatter = figure(width=400, height=350, title=f"{title_prefix} | Regression (r={corr:.2f})",
                           x_axis_label='True SBP (mmHg)', y_axis_label='Predicted SBP (mmHg)')

        # 绘制参考线 y=x (理想情况)
    min_val = min(true_bp.min(), pred_bp.min()) - 5
    max_val = max(true_bp.max(), pred_bp.max()) + 5
    p_scatter.line([min_val, max_val], [min_val, max_val], line_color='black', line_dash='dashed', line_width=2,
                       legend_label="Ideal")

        # 绘制散点
    p_scatter.circle(true_bp, pred_bp, size=6, color=color, alpha=0.6)
    p_scatter.legend.location = "top_left"
    figure_settings(p_scatter)

        # --- 图 2: Bland-Altman 图 ---
        # 作用：展示系统性偏差和一致性。点应在红线范围内且Bias接近0。
    avg_bp = (true_bp + pred_bp) / 2
    diff_bp = pred_bp - true_bp  # Error

        # 计算 LoA (Limits of Agreement)
    upper_loa = bias + 1.96 * std_diff
    lower_loa = bias - 1.96 * std_diff

    p_ba = figure(width=400, height=350, title=f"Bland-Altman (Bias={bias:.2f})",
                      x_axis_label='Average BP (mmHg)', y_axis_label='Error (Pred - True)')

        # 绘制 0 线和 Bias 线
    p_ba.ray(x=[avg_bp.min()], y=[0], length=0, angle=0, line_color="gray", line_dash="solid")
    p_ba.ray(x=[avg_bp.min()], y=[bias], length=0, angle=0, line_color="black", line_width=2, legend_label='Bias')

        # 绘制 LoA 线 (95% 置信区间)
    p_ba.ray(x=[avg_bp.min()], y=[upper_loa], length=0, angle=0, line_color="red", line_dash="dashed",
                 legend_label='+1.96SD')
    p_ba.ray(x=[avg_bp.min()], y=[lower_loa], length=0, angle=0, line_color="red", line_dash="dashed")

        # 绘制散点
    p_ba.circle(avg_bp, diff_bp, size=6, color=color, alpha=0.6)
    figure_settings(p_ba)

        # --- 图 3: 原始时间序列 (保留作为参考，方便看整体走势) ---
    p_ts = figure(width=850, height=250, title=f"Time Series (RMSE={rmse:.2f})",
                      x_axis_label='Heart Beat Index', y_axis_label='SBP (mmHg)')
    p_ts.line(range(len(true_bp)), true_bp, color='black', line_dash='dashed', legend_label='True BP', line_width=2)
    p_ts.scatter(range(len(pred_bp)), pred_bp, color=color, legend_label='Pred', size=4)
    figure_settings(p_ts)
    p_ts.x_range.range_padding = 0.05

    # 返回布局：上面两个并排，下面一个长图
    return column(row(p_scatter, p_ba), p_ts)


if __name__ == "__main__":
    # base_finetuning()
    # pinnfreq_finetuning()
    # pinnfreqct_finetuning()
    # ==========================
    # 加载已微调的 base
    # ==========================
    print("\n>>> 正在加载微调后的 base 模型...")
    pinn_path = os.path.join(MODEL_SAVE_DIR, "PINN_finetuned.h5")
    if not os.path.exists(pinn_path):
        sys.exit(f"错误：找不到模型文件 {pinn_path}，请先运行训练代码！")

    # compile=False 表示加载时不编译优化器和损失函数，仅用于预测
    model_base= tf.keras.models.load_model(pinn_path, compile=False)
    print("base 加载成功。")

    # ==========================
    # 加载已微调的 PINNfreq
    # ==========================
    print("\n>>> 正在加载微调后的 PINNfreq 模型...")
    pinn_path = os.path.join(MODEL_SAVE_DIR, "PINNfreq_finetuned.h5")
    if not os.path.exists(pinn_path):
        sys.exit(f"错误：找不到模型文件 {pinn_path}，请先运行训练代码！")

    # compile=False 表示加载时不编译优化器和损失函数，仅用于预测
    model_pinn = tf.keras.models.load_model(pinn_path, compile=False)
    print("PINNfreq 加载成功。")

    # ==========================
    # 加载已微调的 PINNfreqCT
    # ==========================
    print("\n>>> 正在加载微调后的 PINNfreqCT 模型...")
    ct_path = os.path.join(MODEL_SAVE_DIR, "PINNfreqCT_finetuned.h5")
    if not os.path.exists(ct_path):
        sys.exit(f"错误：找不到模型文件 {ct_path}，请先运行训练代码！")
    # 注意：PINNfreqCT 有自定义层，如果是简单预测，compile=False 通常能避开自定义 Loss 的报错
    model_ct = tf.keras.models.load_model(ct_path, compile=False)
    print("PINNfreqCT 加载成功。")
    print("\n生成结果...")


###绘制预测和真实值的曲线图
    true1 = df_test1['sys'].values
    p_base1, r_base1 = evaluate(model_base, test1_in, true1, "base")
    p_pinn1, r_pinn1 = evaluate(model_pinn, test1_in, true1, "PINNfreq")
    p_ct1, r_ct1 = evaluate(model_ct, test1_in, true1, "PINNfreqCT")


    true2 = df_test2['sys'].values
    p_base2, r_base2 = evaluate(model_base, test2_in, true2, "base")
    p_pinn2, r_pinn2 = evaluate(model_pinn, test2_in, true2, "PINNfreq")
    p_ct2, r_ct2 = evaluate(model_ct, test2_in, true2, "PINNfreqCT")

    plot1 = make_plot(true1, p_base1, p_pinn1, p_ct1, "Exp2: S1D4 (Fine-tuned)", r_base1, r_pinn1, r_ct1)
    plot2 = make_plot(true2, p_base2, p_pinn2, p_ct2, "Exp3: S2D1 (Fine-tuned)", r_base2, r_pinn2, r_ct2)

    save_svg_and_pdf(plot1, "TimeSeries_TimeGen_Compare")
    save_svg_and_pdf(plot2, "TimeSeries_PerGen_Compare")
    # save(column(plot1, plot2), filename=os.path.join(FIG_DIR, "Finetune_Results.html"))
    print(f"结果已保存至 {os.path.join(FIG_DIR, 'TimeSeries_Time/PerGen_Compare.svg')}")

    # print(">>> 正在绘制综合分析图...")
    # # 实验 2 可视化
    # layout_exp2_base = plot_comprehensive_analysis(true1, p_base1, "Exp2 (PINN)", palettes.Colorblind8[5])
    # layout_exp2_pinn = plot_comprehensive_analysis(true1, p_pinn1, "Exp2 (PINNfreq)", palettes.Colorblind8[4])
    # layout_exp2_ct = plot_comprehensive_analysis(true1, p_ct1, "Exp2 (PINNfreqCT)", palettes.Colorblind8[3])
    #
    # # 实验 3 可视化
    # layout_exp3_base = plot_comprehensive_analysis(true2, p_base2, "Exp3 (PINN)", palettes.Colorblind8[5])
    # layout_exp3_pinn = plot_comprehensive_analysis(true2, p_pinn2, "Exp3 (PINNfreq)", palettes.Colorblind8[4])
    # layout_exp3_ct = plot_comprehensive_analysis(true2, p_ct2, "Exp3 (PINNfreqCT)", palettes.Colorblind8[3])
    #
    # # 组合所有图表
    # final_layout = column(
    #     figure(title="=== Exp 2: Time Generalization (S1D4) ===", height=50, width=800, toolbar_location=None),
    #     layout_exp2_base,
    #     layout_exp2_pinn,
    #     layout_exp2_ct,
    #     figure(title="=== Exp 3: Person Generalization (S2D1) ===", height=50, width=800, toolbar_location=None),
    #     layout_exp3_base,
    #     layout_exp3_pinn,
    #     layout_exp3_ct
    # )
    #
    # save_path = os.path.join(FIG_DIR, "Comprehensive_Analysis.html")
    # save(final_layout, filename=save_path)
    # print(f"综合分析报告已保存至: {save_path}")
