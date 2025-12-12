import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn import preprocessing
import os
import sys
import random
import matplotlib.pyplot as plt
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column, row
from bokeh import palettes

# --- 0. 环境与路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

print(f"脚本位置: {SCRIPT_DIR}")
print(f"项目根目录: {PROJECT_ROOT}")

try:
    from model.PINNfreqCT import temporal_contrastive_loss, contextual_contrastive_loss
    from TFdata import compute_stft
    from miscFun import set_global_determinism

    print("成功导入依赖库。")
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)


# --- 1. 全局配置 ---
def set_global_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_global_seeds(42)

DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, 'fig')
MODEL_LOAD_DIR = os.path.join(PROJECT_ROOT, 'saved', 'model')  # 原模型路径
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, 'model')  # 新模型保存路径
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

FILES = {
    'TRAIN': {'bioz': os.path.join(DATA_DIR, 'sub1_day1_bioz.csv'), 'bp': os.path.join(DATA_DIR, 'sub1_day1_BP.csv')},
    'TEST_TIME': {'bioz': os.path.join(DATA_DIR, 'sub1_day4_bioz.csv'),
                  'bp': os.path.join(DATA_DIR, 'sub1_day4_BP.csv')},
    'TEST_PERSON': {'bioz': os.path.join(DATA_DIR, 'sub2_day1_bioz.csv'),
                    'bp': os.path.join(DATA_DIR, 'sub2_day1_BP.csv')}
}

RAW_FS = 1250
# 注意：TARGET_T 将在加载模型后自动更新
EPOCHS = 500  # 微调轮数不需要太多
BATCH_SIZE = 16  # 小数据用小 Batch
LR_FINETUNE = 1e-4  # 微调使用更小的学习率


# --- 2. 自动探测模型维度 ---
def get_model_input_shape(model_path):
    print(f"正在探测模型输入维度: {model_path}")
    try:
        # 加载模型 (只需结构)
        model = tf.keras.models.load_model(model_path, compile=False)
        # PINNfreq 输入列表: [beat, f1, f2, f3]
        # beat shape: (None, T, F)
        input_shape = model.input[0].shape
        T = input_shape[1]
        F = input_shape[2]
        print(f"  > 检测到模型输入 T={T}, F={F}")
        return T, F
    except Exception as e:
        print(f"  > 无法自动检测维度 ({e})，使用默认值 T=34")
        return 34, 18


# 获取 saved/model/PINNfreq.h5 的维度
TARGET_T, TARGET_F = get_model_input_shape(os.path.join(MODEL_LOAD_DIR, "PINNfreq.h5"))


# --- 3. 数据处理流水线 ---
class DataPipeline:
    def __init__(self):
        self.scaler_beats = preprocessing.StandardScaler()
        self.scaler_stft = preprocessing.StandardScaler()
        self.scaler_feat = [preprocessing.StandardScaler() for _ in range(3)]
        self.scaler_out = preprocessing.StandardScaler()

    def load_and_process_csv(self, bioz_path, bp_path):
        if not os.path.exists(bioz_path):
            print(f"错误: 找不到文件 {bioz_path}")
            sys.exit(1)

        print(f"处理: {os.path.basename(bioz_path)} ...")
        df_bioz = pd.read_csv(bioz_path)
        df_bp = pd.read_csv(bp_path)

        t_start = max(df_bioz['time'].min(), df_bp['time'].min())
        t_end = min(df_bioz['time'].max(), df_bp['time'].max())
        mask = (df_bioz['time'] >= t_start) & (df_bioz['time'] <= t_end)

        bioz_time = df_bioz.loc[mask, 'time'].values
        bioz_raw = df_bioz.loc[mask, 'BioZ1'].values

        f_bp = interp1d(df_bp['time'], df_bp['FinapresBP'], kind='linear', fill_value="extrapolate")
        bp_sync = f_bp(bioz_time)

        # 滤波
        sos = signal.butter(4, 0.5, 'hp', fs=RAW_FS, output='sos')
        bioz_filt = signal.sosfiltfilt(sos, bioz_raw)

        # 峰值检测 (阈值稍微降低以防漏检)
        peaks, _ = signal.find_peaks(bp_sync, height=40, distance=int(RAW_FS * 0.4))

        data_list = []
        seg_samples = int(0.8 * RAW_FS)

        for i, p in enumerate(peaks):
            if p - seg_samples < 0: continue

            beat_wave = bioz_filt[p - seg_samples: p]
            sys_val = bp_sync[p]

            # --- 关键：重采样到 TARGET_T (34) ---
            beat_resampled = signal.resample(beat_wave, TARGET_T)

            # 特征提取
            u1 = np.max(beat_wave) - np.min(beat_wave)
            u2 = 5.0  # 简化
            u3 = 75.0  # 简化

            # STFT 并重采样到 (TARGET_T, 17)
            # 原始 TFdata 是 (17, T_raw). 我们需要插值到 (17, TARGET_T)
            # 最终 combine 需要 freq为 (TARGET_T, 17) (转置后)
            # 这里的逻辑需要匹配 TFdata.combine_time_freq

            stft_raw = compute_stft(beat_wave, nfft=32, hoplength=8)  # (17, T_step)
            # 插值 stft 到 TARGET_T 长度
            F_bins = stft_raw.shape[0]  # 17
            stft_res = np.zeros((F_bins, TARGET_T))
            old_x = np.linspace(0, 1, stft_raw.shape[1])
            new_x = np.linspace(0, 1, TARGET_T)
            for f in range(F_bins):
                stft_res[f, :] = interp1d(old_x, stft_raw[f, :], kind='linear', fill_value="extrapolate")(new_x)

            # STFT 转置为 (TARGET_T, 17) 以便 combine
            # 注意: combine_time_freq 内部是 np.concatenate([time, freq], axis=0).T
            # time 是 (1, T), freq 是 (F, T). concat -> (1+F, T). .T -> (T, 1+F)
            # 所以这里我们保存 (F, T) 格式即可

            data_list.append({
                'beat': beat_resampled,  # (T,)
                'stft': stft_res,  # (F, T)
                'sys': sys_val,
                'u1': u1, 'u2': u2, 'u3': u3
            })

        return pd.DataFrame(data_list)

    def fit_scalers(self, df_train):
        print("基于 Train 数据拟合 Scaler...")
        self.scaler_beats.fit(np.stack(df_train['beat'].values).flatten()[:, None])

        all_stft = np.stack(df_train['stft'].values)  # (N, F, T)
        N, F, T = all_stft.shape
        # 对每个频带 fit
        self.scaler_stft.fit(np.transpose(all_stft, (0, 2, 1)).reshape(-1, F))

        self.scaler_feat[0].fit(df_train[['u1']])
        self.scaler_feat[1].fit(df_train[['u2']])
        self.scaler_feat[2].fit(df_train[['u3']])
        self.scaler_out.fit(df_train[['sys']])

    def transform(self, df):
        # Beat
        beats = np.stack(df['beat'].values)
        N, T = beats.shape
        beats_sc = self.scaler_beats.transform(beats.flatten()[:, None]).reshape(N, T)

        # STFT
        stft = np.stack(df['stft'].values)  # (N, F, T)
        N, F, T = stft.shape
        stft_T = np.transpose(stft, (0, 2, 1)).reshape(-1, F)
        stft_sc = self.scaler_stft.transform(stft_T).reshape(N, T, F)  # (N, T, F)

        # STFT (N, T, F) vs Beat (N, T, 1)
        # combine -> (N, T, F+1)
        # 注意：TFdata 的 combine 顺序是 [time, freq].
        # beat (time) 在第0维?
        # TFdata: np.concatenate([time, freq], axis=0).T
        # time=(1, T), freq=(F, T). concat=(1+F, T). T=(T, 1+F).
        # 所以第 0 列是 time (beat)，后面是 freq.
        # 这里 beats_sc 是 (N, T), stft_sc 是 (N, T, F).
        # axis=2 concat. -> (N, T, 1+F)
        X_comb = np.concatenate([beats_sc[:, :, None], stft_sc], axis=2)

        u1 = self.scaler_feat[0].transform(df[['u1']])
        u2 = self.scaler_feat[1].transform(df[['u2']])
        u3 = self.scaler_feat[2].transform(df[['u3']])
        y = self.scaler_out.transform(df[['sys']])

        return [
            tf.convert_to_tensor(X_comb, dtype=tf.float32),
            tf.convert_to_tensor(u1, dtype=tf.float32),
            tf.convert_to_tensor(u2, dtype=tf.float32),
            tf.convert_to_tensor(u3, dtype=tf.float32)
        ], tf.convert_to_tensor(y, dtype=tf.float32)

pipeline = DataPipeline()

# 加载数据
df_train = pipeline.load_and_process_csv(FILES['TRAIN']['bioz'], FILES['TRAIN']['bp'])
df_test1 = pipeline.load_and_process_csv(FILES['TEST_TIME']['bioz'], FILES['TEST_TIME']['bp'])
df_test2 = pipeline.load_and_process_csv(FILES['TEST_PERSON']['bioz'], FILES['TEST_PERSON']['bp'])
print(f"数据量: Train={len(df_train)}, Test1={len(df_test1)}, Test2={len(df_test2)}")

pipeline.fit_scalers(df_train)
train_in, train_y = pipeline.transform(df_train)
test1_in, test1_y = pipeline.transform(df_test1)
test2_in, test2_y = pipeline.transform(df_test2)

def base_finetuning():
    # ==========================
    # 微调 PINNfreq
    # ==========================
    print("\n>>> 加载并微调 PINN...")
    model_base = tf.keras.models.load_model(os.path.join(MODEL_LOAD_DIR, "PINN.h5"))
    optimizer_base = tf.keras.optimizers.Adam(LR_FINETUNE)
    # 准备 PINN 数据
    # train_in 是个 list，直接传会被打包报错。转为 tuple 则会被视为多个独立的输入。
    ds_pinn = tf.data.Dataset.from_tensor_slices((
        tuple([tf.expand_dims(train_in[0][:,:,0], axis=-1),train_in[1],train_in[2],train_in[3]]),  # <--- 关键修改：变成元组 (beat, u1, u2, u3)
        train_y
    )).shuffle(100).batch(BATCH_SIZE, drop_remainder=True)

    for epoch in range(EPOCHS):
        total_loss = 0
        steps = 0
        # 解包时也要注意
        for inputs, target in ds_pinn:
            # inputs 现在是一个元组 (beat, u1, u2, u3)
            # 我们把它转回 list 传给模型 (因为模型 call 期望 list)
            inputs_list = list(inputs)

            # 手动提取各个部分用于物理计算
            inp_beat, inp_f1, inp_f2, inp_f3 = inputs

            with tf.GradientTape(persistent=True) as tape:
                tape.watch([inp_f1, inp_f2, inp_f3])  # 监视物理特征

                # 前向传播
                y_pred = model_base(inputs_list, training=True)

                # 1. MSE Loss
                loss_mse = K.mean(K.square(y_pred - target))


                # 物理损失 (简化: 仅当 batch > 1)
                if tf.shape(y_pred)[0] > 1:
                    # 简单的差分约束
                    loss_phy = K.mean(K.square((y_pred[1:] - y_pred[:-1]))) * 0.1
                else:
                    loss_phy = 0.0

                loss = loss_mse + loss_phy

            grads = tape.gradient(loss, model_base.trainable_weights)
            optimizer_base.apply_gradients(zip(grads, model_base.trainable_weights))
            total_loss += loss
            steps += 1

        if (epoch + 1) % 50 == 0:
            print(f"  base Epoch {epoch + 1} Loss: {total_loss / steps:.4f}")

    model_base.save(os.path.join(MODEL_SAVE_DIR, "PINN_finetuned.h5"))

# --- 4. 迁移学习流程 ---
def pinnfreq_finetuning():
    # ==========================
    # 微调 PINNfreq
    # ==========================
    print("\n>>> 加载并微调 PINNfreq ...")
    model_pinn = tf.keras.models.load_model(os.path.join(MODEL_LOAD_DIR, "PINNfreq.h5"))
    optimizer_pinn = tf.keras.optimizers.Adam(LR_FINETUNE)

    # train_in 是个 list，直接传会被打包报错。转为 tuple 则会被视为多个独立的输入。
    ds_pinn = tf.data.Dataset.from_tensor_slices((
        tuple(train_in),  # <--- 关键修改：变成元组 (beat, u1, u2, u3)
        train_y
    )).shuffle(100).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        total_loss = 0
        steps = 0
        # 解包时也要注意
        for inputs, target in ds_pinn:
            # inputs 现在是一个元组 (beat, u1, u2, u3)
            # 我们把它转回 list 传给模型 (因为模型 call 期望 list)
            inputs_list = list(inputs)

            # 手动提取各个部分用于物理计算
            inp_beat, inp_f1, inp_f2, inp_f3 = inputs

            with tf.GradientTape(persistent=True) as tape:
                tape.watch([inp_f1, inp_f2, inp_f3])  # 监视物理特征

                # 前向传播
                y_pred = model_pinn(inputs_list, training=True)

                # 1. MSE Loss
                loss_mse = K.mean(K.square(y_pred - target))

                # 物理损失 (简化: 仅当 batch > 1)
                if tf.shape(y_pred)[0] > 1:
                    # 简单的差分约束
                    loss_phy = K.mean(K.square((y_pred[1:] - y_pred[:-1]))) * 0.1
                else:
                    loss_phy = 0.0

                loss = loss_mse + loss_phy

                # 总 Loss
                loss = loss_mse + loss_phy

            grads = tape.gradient(loss, model_pinn.trainable_weights)
            optimizer_pinn.apply_gradients(zip(grads, model_pinn.trainable_weights))
            total_loss += loss
            steps += 1

        if (epoch + 1) % 50 == 0:
            print(f"  PINNfreq Epoch {epoch + 1} Loss: {total_loss / steps:.4f}")

    model_pinn.save(os.path.join(MODEL_SAVE_DIR, "PINNfreq_finetuned.h5"))

def pinnfreqct_finetuning():
    # ==========================
    # 微调 PINNfreqCT
    # ==========================
    print("\n>>> 加载并微调 PINNfreqCT ...")
    # PINNfreqCT 加载可能需要 custom_objects
    try:
        model_ct = tf.keras.models.load_model(os.path.join(MODEL_LOAD_DIR, "PINNfreqCT.h5"), compile=False)
    except:
        print("直接加载失败，尝试重建模型结构并加载权重...")
        # 如果保存的是整个模型但加载出错，这里需要重建结构。
        # 暂时假设能加载成功
        sys.exit("PINNfreqCT 模型加载失败")

    optimizer_ct = tf.keras.optimizers.Adam(LR_FINETUNE)

    # 构造数据 (Beat_W = Beat_S)
    ds_ct = tf.data.Dataset.from_tensor_slices((
        (train_in[0], train_in[0], train_in[1], train_in[2], train_in[3]),
        train_y
    )).batch(BATCH_SIZE)

    print("开始 PINNfreqCT 微调循环...")
    for epoch in range(EPOCHS):
        total_loss = 0
        steps = 0
        for inputs, target in ds_ct:
            with tf.GradientTape() as tape:
                # inputs: [beat_w, beat_s, u1, u2, u3]
                outs = model_ct(list(inputs), training=True)
                y_nn_w = outs[0]
                Z_w, Z_s = outs[2], outs[3]
                C_w, C_s = outs[4], outs[5]

                l_mse = K.mean(K.square(y_nn_w - target))
                l_tc = temporal_contrastive_loss(Z_w, Z_s)
                l_cc = contextual_contrastive_loss(C_w, C_s)

                # 物理损失 (简化: 仅当 batch > 1)
                if tf.shape(y_nn_w)[0] > 1:
                    # 简单的差分约束
                    loss_phy = K.mean(K.square((y_nn_w[1:] - y_nn_w[:-1]))) * 0.1
                else:
                    loss_phy = 0.0

                loss = l_mse + 0.5 * l_tc + 0.5 * l_cc + loss_phy

            grads = tape.gradient(loss, model_ct.trainable_weights)
            optimizer_ct.apply_gradients(zip(grads, model_ct.trainable_weights))
            total_loss += loss
            steps += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch + 1} Loss: {total_loss / steps:.4f}")

    model_ct.save(os.path.join(MODEL_SAVE_DIR, "PINNfreqCT_finetuned.h5"))