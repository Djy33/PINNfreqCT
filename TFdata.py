from miscFun import *
import numpy as np
import pandas as pd
import librosa
from scipy.interpolate import interp1d


def compute_stft(beat, nfft, hoplength):
    #nfft采样点，hoplength窗口滑动的步长
    beat = np.array(beat, dtype=np.float32)#列表转numpy
    stft_mag = np.abs(librosa.stft(beat, n_fft=nfft, hop_length=hoplength))
    #log幅值,1e-6防止log0
    stft_log = np.log(stft_mag + 1e-6)
    # 2D 矩阵 (freq × time)
    return stft_log

def insert_num(freq, time):#一维先线性插值确保freq的时间维度和time保持一致
    freq_bins, time_frames = freq.shape
    T = len(time)
    #原x轴
    old_time = np.linspace(0,T-1,time_frames)
    new_time = np.arange(T)
    #重新插值
    stft_resized = np.zeros((freq_bins,T),dtype=np.float32)

    for i in range(freq_bins):
        f = interp1d(old_time, freq[i,:], kind='linear', bounds_error=False, fill_value=(freq[i,0],freq[i,-1]))
        stft_resized[i, :] = f(new_time)
    return stft_resized

def combine_time_freq(time, freq):
    # time(T,)freq(F,T)
    time = np.expand_dims(time, axis=0)#(1,T)
    combined = np.concatenate([time, freq], axis=0) #(1+F,T)
    #返回转置，（T,1+F)
    return combined.T


#一个心跳周期的完整波形（BIOZ/ECG/PPG） → 对应一个收缩压（Sys）
df_demo_data = pd.read_pickle('../data_demo_pinn_bioz_bp',compression='gzip')#noise.py运行时
# df_demo_data = pd.read_pickle('data_demo_pinn_bioz_bp',compression='gzip')#main.py运行时
df_demo_data['bioz_stft'] = df_demo_data['bioz_beats'].apply(
    lambda x: compute_stft(x, nfft=32, hoplength=8)
)
df_demo_data['bioz_stft_insert'] = df_demo_data.apply(
    lambda x: insert_num(x['bioz_stft'], x['bioz_beats']),
    axis=1
)
# 列表推导式实现“逐行转置”(time×freq)
stft_T = [np.array(stft).T for stft in df_demo_data['bioz_stft_insert']]
#预处理和准备 Train/Test Datasets
# 初始化种子，可复现
SEED = 123
set_global_determinism(seed=SEED)
# The keys for the beat data (beat_key), the target (out_key), and the features (feat_keys) are defined
beat_key = 'bioz_beats'
#收缩压
out_key = 'sys'
feat_keys = ['phys_feat_1','phys_feat_2','phys_feat_3']
# Data scaling of BP, input beats, and input features
# This scaler standardizes by removing the mean and scaling to unit variance
# This is done to ensure having the same scale, which can improve the performance of machine learning algorithms
scaler_out = preprocessing.StandardScaler().fit(df_demo_data[out_key].to_numpy()[:, None])
scaler_beats = preprocessing.StandardScaler().fit(np.concatenate(df_demo_data[beat_key].to_numpy())[:, None])
scaler_stft = preprocessing.StandardScaler().fit(np.concatenate(stft_T))
scaler_X = [preprocessing.StandardScaler().fit(df_demo_data[a].to_numpy()[:, None]) for a in feat_keys]
# Apply Scaling，标准化
# The scaled versions of the BP, input beats, and input features are then added to the dataframe
# axis=1 = 对 DataFrame 逐行 处理
df_demo_data.loc[df_demo_data.index,'bioz_beats_scaled'] = df_demo_data.apply(lambda x: np.concatenate(scaler_beats.transform(x['bioz_beats'][:, None])), axis=1).to_numpy()
#标准化器的要求：sklearn 的 StandardScaler/MinMaxScaler 仅接收 (n_samples, n_features) 的 2 维输入（样本数 × 特征数），不支持 1 维或 3 维。
df_demo_data.loc[df_demo_data.index,'bioz_stft_scaled'] = df_demo_data.apply( lambda x: scaler_stft.transform(np.array(x['bioz_stft_insert']).T).T, axis=1)#转置→标准化→转回逻辑,转回原 (F, T)
df_demo_data.loc[df_demo_data.index,out_key+'_scaled'] = df_demo_data.apply(lambda x: np.concatenate(scaler_out.transform(np.array([x[out_key]])[:, None]))[0], axis=1).to_numpy()
for tmp_key, tmp_count in zip(feat_keys, range(len(feat_keys))):
    df_demo_data.loc[df_demo_data.index, tmp_key+'_scaled'] = df_demo_data.apply(lambda x: np.concatenate(scaler_X[tmp_count].transform(np.array([x[tmp_key]])[:, None])), axis=1).to_numpy()
# print(df_demo_data)
#（T,1+F) （34，18）
df_demo_data['time&freq'] = df_demo_data.apply(
    lambda row: combine_time_freq(
        np.array(row['bioz_beats_scaled']),
        np.array(row['bioz_stft_scaled']),
    ),
    axis=1,
)
# Fetch scaled feature names
X_keys = [a+'_scaled' for a in feat_keys]
# Prepare train/test using minimal training the BP
# Fetch data shapes，一个心跳周期的时序数据输入长度、输出序列长度
# length_seq_x = df_demo_data.apply(lambda x: len(x[beat_key+'_scaled']), axis=1).unique()[0]
length_seq_x = df_demo_data.apply(lambda x: len(x['time&freq']), axis=1).unique()[0]
# print(length_seq_x)
# Set the length of the target to 1
length_seq_y = 1
# Start with all points
# Reshape the scaled beat data into a 2D array where each row corresponds to a sample and each column corresponds to a time point in the beat sequence
# The same is done for the features and the target类似于提取'bioz_beats'这一列，并为大列表[[.....],[.....],...]
all_beats = np.stack(df_demo_data['time&freq'].values,axis=0)
# print(all_beats.shape)
[all_feat1, all_feat2, all_feat3] = [df_demo_data[a].values[:, None] for a in X_keys]
all_out = df_demo_data[out_key+'_scaled'].values[:, None]
# Used only for plotting purposes，scaler_out.inverse_transform()把标准化数据恢复成“真实血压值”。
out_max_rescaled = np.concatenate(scaler_out.inverse_transform(all_out[:, 0][:, None])).max()
out_min_rescaled = np.concatenate(scaler_out.inverse_transform(all_out[:, 0][:, None])).min()
# Given different trials have time gaps, ignore first 3 instances from indices to prevent discontiunity in training
list_all_length = [0]
for _, df_tmp in df_demo_data.groupby(['trial_id']):
    list_all_length.append(len(df_tmp))
#np.arange(start, stop, step)np.cumsum计算每个id的起始位置，并去掉所有数据的结束位置
ix_ignore_all = np.concatenate(np.array([np.arange(a, a+3,1) for a in list(np.cumsum(list_all_length)[:-1])]))
# Update the final indices set
ix_all=list(set(np.arange(len(df_demo_data)))-set(ix_ignore_all))
# Separate train/test based on minimal training criterion
random.seed(0)
bp_dist = df_demo_data[out_key].values

# Find indices for train and test datasets
# The target values are sorted in ascending order, and the sorted indices are split into multiple subsets
# For each subset, a random index is selected as a training index
#步长 1 划分区间,用 histogram 按数值区间切分数据，论文规定：一个心跳周期的幅值为40，则训练的采样点为40
ix_split = np.split([a for a in np.argsort(bp_dist) if a not in set(ix_ignore_all)], np.cumsum(np.histogram(bp_dist[ix_all],bins=np.arange(bp_dist[ix_all].min(), bp_dist[ix_all].max(), 1))[0]))
ix_train = [random.Random(4).choice(a) if len(a)>0 else -1 for a in ix_split]
ix_train = list(set(ix_train)-set([-1]))

# Test set is all remaining points not used for training
ix_test = list(set(ix_all) - set(ix_train))

# Build train and test datasets based on the indices
train_beats = all_beats[ix_train, :,:]
# print(train_beats.shape)
test_beats = all_beats[ix_test, :,:]
[train_feat1, train_feat2, train_feat3] = [all_feat1[ix_train, :], all_feat2[ix_train, :], all_feat3[ix_train, :]]
[test_feat1, test_feat2, test_feat3] = [all_feat1[ix_test, :], all_feat2[ix_test, :], all_feat3[ix_test, :]]
train_out = all_out[ix_train, :]#目标值如收缩压
test_out = all_out[ix_test, :]