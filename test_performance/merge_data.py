import pandas as pd
import numpy as np
import os
import sys

# --- 配置部分 ---
# 定义你希望如何合并数据
# 格式: "目标文件名": ["原始文件1", "原始文件2", ...]
# 注意：这里假设你已经把所有 BioZ 和 BP 文件都放在了 raw_data 文件夹下
# 并且 BioZ 和 BP 的文件名后缀是固定的 (_bioz.csv 和 _finapresBP.csv)

MERGE_CONFIG = {
    # 训练集：Subject 1 Day 1 (拼接 Trial 01, 03, 04, 05, 06, 07)
    # 注意：我假设 sub1_day1_bioz.csv 其实就是 data_trial01
    # 如果你本地文件名是 data_trial01_bioz.csv，请相应修改列表
    "sub1_day1": [
        "data_trial03",
        "data_trial04",
        "data_trial05",
        "data_trial06",
        "data_trial07"
    ],

    # # 测试集 1：Subject 1 Day 4 (如果有多个 trial，请在列表中继续添加)
    "sub1_day4": [
        "data_trial01",
        "data_trial02",
        "data_trial03",
        "data_trial04",
        "data_trial05"
    ],
    #
    # # 测试集 2：Subject 2 Day 1
    "sub2_day1": [
        "data_trial03",
        "data_trial04",
        "data_trial05",
        "data_trial06",
        "data_trial07"
    ],
}

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_data")  # 请建立这个文件夹放原始数据
OUTPUT_DIR = os.path.join(BASE_DIR, "data")  # 输出到 per-day.py 使用的 data 目录

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def merge_csv_group(target_name, source_prefixes):
    print(f"\n正在合并任务: {target_name} ...")

    all_bioz = []
    all_bp = []

    current_time_offset = 0.0
    gap = 1.0  # 两个 trial 之间插入 1秒 的间隔

    # 遍历列表中的每个文件前缀
    for prefix in source_prefixes:
        # 构造完整文件名 (支持 .csv)
        # 逻辑：自动寻找 raw_data 下的 _bioz.csv 和 _finapresBP.csv (或 _BP.csv)

        # 1. 寻找 BioZ 文件
        bioz_candidates = [
            f"{prefix}_bioz.csv",
            f"{prefix}.csv"  # 有些可能没有后缀
        ]
        bioz_path = None
        for cand in bioz_candidates:
            p = os.path.join(RAW_DIR, target_name,cand)
            if os.path.exists(p):
                bioz_path = p
                break

        # 2. 寻找 BP 文件
        # 注意：你的文件名有时是 _finapresBP.csv，有时是 _BP.csv
        bp_candidates = [
            f"{prefix}_finapresBP.csv",
            f"{prefix}_BP.csv"
        ]
        bp_path = None
        for cand in bp_candidates:
            p = os.path.join(RAW_DIR,target_name, cand)
            if os.path.exists(p):
                bp_path = p
                break

        if not bioz_path or not bp_path:
            print(f"  [跳过] 找不到完整文件对: {prefix} (BioZ: {bioz_path}, BP: {bp_path})")
            continue

        print(f"  + 读取片段: {os.path.basename(bioz_path)} & {os.path.basename(bp_path)}")

        # 读取数据
        df_bioz = pd.read_csv(bioz_path)
        df_bp = pd.read_csv(bp_path)

        # 确保按时间排序
        df_bioz = df_bioz.sort_values('time')
        df_bp = df_bp.sort_values('time')

        # 获取当前片段的时间范围
        t_min = min(df_bioz['time'].min(), df_bp['time'].min())
        t_max = max(df_bioz['time'].max(), df_bp['time'].max())
        duration = t_max - t_min

        # --- 核心：重写时间戳 ---
        # 新时间 = (原时间 - 原开始时间) + 当前累计偏移量
        df_bioz['time'] = (df_bioz['time'] - t_min) + current_time_offset
        df_bp['time'] = (df_bp['time'] - t_min) + current_time_offset

        # 添加到列表
        all_bioz.append(df_bioz)
        all_bp.append(df_bp)

        # 更新偏移量 (为下一个文件做准备)
        current_time_offset += (duration + gap)

    if not all_bioz:
        print(f"  错误: {target_name} 没有合并任何数据！")
        return

    # 合并 DataFrame
    merged_bioz = pd.concat(all_bioz, ignore_index=True)
    merged_bp = pd.concat(all_bp, ignore_index=True)

    # 保存结果
    out_bioz = os.path.join(OUTPUT_DIR, f"{target_name}_bioz.csv")
    out_bp = os.path.join(OUTPUT_DIR, f"{target_name}_BP.csv")

    merged_bioz.to_csv(out_bioz, index=False)
    merged_bp.to_csv(out_bp, index=False)

    print(f"  => 合并完成！总时长: {current_time_offset:.2f} 秒")
    print(f"     BioZ: {out_bioz} ({len(merged_bioz)} 行)")
    print(f"     BP:   {out_bp} ({len(merged_bp)} 行)")


if __name__ == "__main__":
    # 检查 raw_data 文件夹
    if not os.path.exists(RAW_DIR):
        print(f"错误: 请先创建 '{RAW_DIR}' 文件夹，并把原始 CSV 文件放进去！")
    else:
        print("开始合并数据...")
        for target, sources in MERGE_CONFIG.items():
            merge_csv_group(target, sources)
        print("\n所有处理结束。现在可以运行 per-day.py 了。")