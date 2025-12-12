from miscFun import *
import pickle
'''
N_INPUT: The number of input bio-z dimensions for one heartbeat 输入维度
N_FEAT: The number of physiological features 生理特征维度
N_EXT: The number of features extracted by the CNN提取特征数量
'''


def model_DNN(N_INPUT, N_FEAT=1, N_EXT=100):
    # The input to the model is a 1D tensor representing a time series of heartbeat data, sampled with 250/8 points for 30 seconds
    inp_beat = tf.keras.Input(shape=(N_INPUT))

    # Define the 1D CNN for NN feature extraction
    # The input tensor is first expanded by one dimension (from 1D to 2D) to be compatible with the Conv1D layer
    # （样本批量数、时序的时间步（采样点）、信号的通道数）
    cnn1_1 = tf.keras.layers.Conv1D(32, 5, activation='relu')(tf.keras.backend.expand_dims(inp_beat, axis=-1))
    cnn1_2 = tf.keras.layers.Conv1D(64, 3, activation='relu')(cnn1_1)
    mp_cnn1 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1)(cnn1_2)
    fl_cnn1 = tf.keras.layers.Flatten()(mp_cnn1)

    # A fully connected layer further processes the flattened tensor and extracts N_EXT features
    feat_ext = tf.keras.layers.Dense(N_EXT, activation='relu')(fl_cnn1)

    # Define physiological features (case study uses 3 features), each of these features is expected to be a 1D tensor
    # (批大小, N_FEAT)
    inp_feat1 = tf.keras.Input(shape=(N_FEAT))  # feat 1
    inp_feat2 = tf.keras.Input(shape=(N_FEAT))  # feat 2
    inp_feat3 = tf.keras.Input(shape=(N_FEAT))  # feat 3

    # The extracted features and physiological features are concatenated together
    feat_comb = tf.keras.layers.Concatenate()([inp_feat1, inp_feat2, inp_feat3, feat_ext])

    # A fully connected layer with is applied to the concatenated features
    dense1_1 = tf.keras.layers.Dense(60, activation='relu')(feat_comb)
    out = tf.keras.layers.Dense(N_FEAT)(dense1_1)

    # Finally, the model is instantiated with the specified inputs and outputs
    model = tf.keras.Model(inputs=[inp_beat, inp_feat1, inp_feat2, inp_feat3], outputs=[out])
    return model
# 在训练完成后保存损失列表
def save_losses(loss_list, test_loss_list, save_path="./saved/loss/losses_CNN.pkl"):
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存数据
    with open(save_path, 'wb') as f:
        pickle.dump({
            'loss_list_conv': loss_list,
            'test_loss_list_conv': test_loss_list
        }, f)
    print(f"损失数据已保存至：{save_path}")

# 加载损失列表
def load_losses(load_path="./saved/loss/losses_CNN.pkl"):
    global loss_list_conv, test_loss_list_conv
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    loss_list_conv = data['loss_list_conv']
    test_loss_list_conv = data['test_loss_list_conv']




# 1. 获取当前代码文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 2. 获取代码文件所在的目录
current_dir = os.path.dirname(current_file_path)
data_path = os.path.join(current_dir, "../data_demo_pinn_bioz_bp")
#一个心跳周期的完整波形（BIOZ/ECG/PPG） → 对应一个收缩压（Sys）
df_demo_data = pd.read_pickle(data_path,compression='gzip')

# Initialize a SEED value to ensure that the random processes in the code can be reproduced.
SEED = 123
# Call the function with seed value
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
scaler_X = [preprocessing.StandardScaler().fit(df_demo_data[a].to_numpy()[:, None]) for a in feat_keys]

# Apply Scaling，标准化
# The scaled versions of the BP, input beats, and input features are then added to the dataframe
# axis=1 = 对 DataFrame 逐行 处理
df_demo_data.loc[df_demo_data.index,beat_key+'_scaled'] = df_demo_data.apply(lambda x: np.concatenate(scaler_beats.transform(x[beat_key][:, None])), axis=1).to_numpy()
df_demo_data.loc[df_demo_data.index,out_key+'_scaled'] = df_demo_data.apply(lambda x: np.concatenate(scaler_out.transform(np.array([x[out_key]])[:, None]))[0], axis=1).to_numpy()
for tmp_key, tmp_count in zip(feat_keys, range(len(feat_keys))):
    df_demo_data.loc[df_demo_data.index, tmp_key+'_scaled'] = df_demo_data.apply(lambda x: np.concatenate(scaler_X[tmp_count].transform(np.array([x[tmp_key]])[:, None])), axis=1).to_numpy()

# Fetch scaled feature names
X_keys = [a+'_scaled' for a in feat_keys]

# Prepare train/test using minimal training the BP
# Fetch data shapes，一个心跳周期的时序数据输入长度、输出序列长度
length_seq_x = df_demo_data.apply(lambda x: len(x[beat_key+'_scaled']), axis=1).unique()[0]
# Set the length of the target to 1
length_seq_y = 1
# Start with all points
# Reshape the scaled beat data into a 2D array where each row corresponds to a sample and each column corresponds to a time point in the beat sequence
# The same is done for the features and the target类似于提取'bioz_beats'这一列，并为大列表[[.....],[.....],...]
all_beats = np.reshape(np.concatenate(df_demo_data[beat_key+'_scaled'].values), (len(df_demo_data), length_seq_x))
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
train_beats = all_beats[ix_train, :]
test_beats = all_beats[ix_test, :]
[train_feat1, train_feat2, train_feat3] = [all_feat1[ix_train, :], all_feat2[ix_train, :], all_feat3[ix_train, :]]
[test_feat1, test_feat2, test_feat3] = [all_feat1[ix_test, :], all_feat2[ix_test, :], all_feat3[ix_test, :]]
train_out = all_out[ix_train, :]#目标值如收缩压
test_out = all_out[ix_test, :]
#### Define model input tensors
# The training, testing, and all data are converted to TensorFlow tensors
# The tensors for the different datasets are grouped into lists

model_inp = tf.convert_to_tensor(train_beats, dtype=tf.float32)
feat1_inp = tf.convert_to_tensor(train_feat1, dtype=tf.float32)
feat2_inp = tf.convert_to_tensor(train_feat2, dtype=tf.float32)
feat3_inp = tf.convert_to_tensor(train_feat3, dtype=tf.float32)
inp_comb = [model_inp, feat1_inp, feat2_inp, feat3_inp]

model_inp_test = tf.convert_to_tensor(test_beats, dtype=tf.float32)
feat1_inp_test = tf.convert_to_tensor(test_feat1, dtype=tf.float32)
feat2_inp_test = tf.convert_to_tensor(test_feat2, dtype=tf.float32)
feat3_inp_test = tf.convert_to_tensor(test_feat3, dtype=tf.float32)
inp_comb_test = [model_inp_test, feat1_inp_test, feat2_inp_test, feat3_inp_test]

model_inp_all = tf.convert_to_tensor(all_beats, dtype=tf.float32)
feat1_inp_all = tf.convert_to_tensor(all_feat1, dtype=tf.float32)
feat2_inp_all = tf.convert_to_tensor(all_feat2, dtype=tf.float32)
feat3_inp_all = tf.convert_to_tensor(all_feat3, dtype=tf.float32)
inp_comb_all = [model_inp_all, feat1_inp_all, feat2_inp_all, feat3_inp_all]
# Two lists are initialized to keep track of the training and testing loss during each epoch
loss_list_conv = []
test_loss_list_conv = []


def cnn():
# A Deep Neural Network model is initialized with the dimension of the beats, the diemnsion of each feature, and the number of neurons in the first dense layer(CNN提取特征数量)
    model_dnn_conv = model_DNN(np.shape(train_beats)[-1], 1, 64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    epochs = 5000
    for epoch in range(epochs):
        with tf.GradientTape() as tape:

            tape.watch(inp_comb)
            # Traditional out
            yh = model_dnn_conv(inp_comb, training=True)  # 模型预测值
            loss_ini = yh - train_out
            loss = K.mean(K.square(loss_ini))

        grads = tape.gradient(loss, model_dnn_conv.trainable_weights)

        loss_list_conv.append(float(loss))
        loss_final = np.min(loss_list_conv)
        optimizer.apply_gradients(zip(grads, model_dnn_conv.trainable_weights))

        pred_out = model_dnn_conv(inp_comb_test)

        test_loss_ini = pred_out - test_out
        test_loss = K.mean(K.square(test_loss_ini))
        test_loss_list_conv.append(float(test_loss))

    # If the training loss reaches a minimum value of 0.01, or the maximum number of epochs is reached, the training process is stopped
        if (loss_final <= 0.01) | (epoch == epochs - 1):
            print("CNN model training Completed. Epoch %d/%d -- loss: %.4f" % (epoch, epochs, float(loss)))
            break

    # The trained model1's predictions on the test dataset are computed
    pred_out = model_dnn_conv(inp_comb_test)

    # The Pearson correlation coefficient and the Root Mean Square Error are calculated between the actual and predicted test outcomes
    corr_conv = np.corrcoef(np.concatenate(test_out)[:], np.concatenate(pred_out)[:])[0][1]
    rmse_conv = np.sqrt(np.mean(np.square
                                (np.concatenate(scaler_out.inverse_transform(np.concatenate(test_out)[:][:, None])) -
                                 np.concatenate(scaler_out.inverse_transform(np.concatenate(pred_out)[:][:, None])))))
    print('#### CNN Performance ####')
    print('Corr: %.2f,  RMSE: %.1f' % (corr_conv, rmse_conv))

    model_save_path = "./saved/model/CNN.h5"  # HDF5格式
    model_dnn_conv.save(model_save_path)
    print(f"模型已保存至：{model_save_path}")

    # 训练结束后保存损失
    save_losses(loss_list_conv, test_loss_list_conv)

    return model_dnn_conv