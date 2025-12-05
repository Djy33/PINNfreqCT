from TFdata import *
import pickle
output_notebook()

# Physics Informed Neural Network with Taylor Series

'''
N_INPUT: The number of input bio-z dimensions for one heartbeat 输入维度
N_FEAT: The number of physiological features 生理特征维度
N_EXT: The number of features extracted by the CNN提取特征数量
'''

def model_DNN(T, F, N_FEAT=1, N_EXT=100):
    # The input to the model1 is a 1D tensor representing a time series of heartbeat data, sampled with 250/8 points for 30 seconds
    inp_beat=tf.keras.Input(shape=(T,F))

    # Define the 1D CNN for NN feature extraction
    # The input tensor is first expanded by one dimension (from 1D to 2D) to be compatible with the Conv1D layer
    #（样本批量数、时序的时间步（采样点）、信号的通道数）
    # cnn1_1 = tf.keras.layers.Conv1D(32,5,activation='relu')(tf.keras.backend.expand_dims(inp_beat,axis=-1))
    cnn1_1 = tf.keras.layers.Conv1D(32,5,activation='relu')(inp_beat)
    cnn1_2 = tf.keras.layers.Conv1D(64,3,activation='relu')(cnn1_1)
    mp_cnn1 = tf.keras.layers.MaxPooling1D(pool_size=3,strides=1)(cnn1_2)
    fl_cnn1 = tf.keras.layers.Flatten()(mp_cnn1)

    # A fully connected layer further processes the flattened tensor and extracts N_EXT features
    feat_ext = tf.keras.layers.Dense(N_EXT,activation='relu')(fl_cnn1)

    # Define physiological features (case study uses 3 features), each of these features is expected to be a 1D tensor
    #(批大小, N_FEAT)
    inp_feat1 = tf.keras.Input(shape=(N_FEAT)) # feat 1
    inp_feat2 = tf.keras.Input(shape=(N_FEAT)) # feat 2
    inp_feat3 = tf.keras.Input(shape=(N_FEAT)) # feat 3

    # The extracted features and physiological features are concatenated together
    feat_comb = tf.keras.layers.Concatenate()([inp_feat1,inp_feat2,inp_feat3,feat_ext])

    # A fully connected layer with is applied to the concatenated features
    dense1_1 = tf.keras.layers.Dense(60,activation='relu')(feat_comb)
    out = tf.keras.layers.Dense(N_FEAT)(dense1_1)

    # Finally, the model1 is instantiated with the specified inputs and outputs
    model = tf.keras.Model(inputs=[inp_beat, inp_feat1, inp_feat2, inp_feat3], outputs=[out])
    return model

#### Define model1 input tensors
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


# 在训练完成后保存损失列表
def save_losses(loss_list, test_loss_list, save_path="./saved/loss/losses_PINNfreq.pkl"):
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存数据
    with open(save_path, 'wb') as f:
        pickle.dump({
            'loss_list_pinn': loss_list,
            'test_loss_list_pinn': test_loss_list
        }, f)
    print(f"损失数据已保存至：{save_path}")

# 加载损失列表
def load_losses(load_path="./saved/loss/losses_PINNfreq.pkl"):
    global loss_list_pinn, test_loss_list_pinn
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    loss_list_pinn = data['loss_list_pinn']
    test_loss_list_pinn = data['test_loss_list_pinn']

# Two lists are initialized to keep track of the training and testing loss during each epoch
loss_list_pinn = []
test_loss_list_pinn = []
def pinn_freq():
#############################
############### PINN MODEL
#############################
# A Deep Neural Network model1 is initialized with the dimension of the beats, the diemnsion of each feature, and the number of neurons in the first dense layer
    T = train_beats.shape[1]
    F = train_beats.shape[2]
    model_dnn_pinn = model_DNN(T, F, 1, 64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)

    epochs = 5000
    for epoch in range(epochs):
        with tf.GradientTape() as tape:

            tape.watch(inp_comb)
            # Traditional out
            yh = model_dnn_pinn(inp_comb, training=True)
            loss_ini = yh - train_out
            loss = K.mean(K.square(loss_ini))

            # Additional tf.GradientTape contexts are used to compute the derivatives of the model1's predictions with respect to the features
            # 用的全部数据inp_comb_all，没有划分训练和测试集
            with tf.GradientTape() as deriv_f1:
                deriv_f1.watch(inp_comb_all)
                yhp = model_dnn_pinn(inp_comb_all, training=True)
            dx_f1 = deriv_f1.gradient(yhp, feat1_inp_all)

            with tf.GradientTape() as deriv_f2:
                deriv_f2.watch(inp_comb_all)
                yhp = model_dnn_pinn(inp_comb_all, training=True)
            dx_f2 = deriv_f2.gradient(yhp, feat2_inp_all)

            with tf.GradientTape() as deriv_f3:
                deriv_f3.watch(inp_comb_all)
                yhp = model_dnn_pinn(inp_comb_all, training=True)
            dx_f3 = deriv_f3.gradient(yhp, feat3_inp_all)

        # A physics-based prediction is computed by adding the model1's predictions to the product of the computed derivatives and
        # the differences in the feature values between consecutive timesteps，-1：最后一行，又因为左包右不包，从[0，倒数第二行]
            pred_physics = (yhp[:-1, 0]
                        + Multiply()([dx_f1[:-1, 0], feat1_inp_all[1:, 0] - feat1_inp_all[:-1, 0]])
                        + Multiply()([dx_f2[:-1, 0], feat2_inp_all[1:, 0] - feat2_inp_all[:-1, 0]])
                        + Multiply()([dx_f3[:-1, 0], feat3_inp_all[1:, 0] - feat3_inp_all[:-1, 0]])
                        )
        # 每个心跳周期都有一个预测值
            physics_loss_ini = pred_physics - yhp[1:, 0]
            physics_loss = K.mean(K.square(tf.gather_nd(physics_loss_ini, indices=np.array(ix_all[:-1])[:, None])))

        # The total loss is computed as the sum of the initial loss and ten times the physics-based loss
        # The physics-based loss is multiplied by a factor of ten to emphasize its importance in the loss function
            loss_total = loss + physics_loss * 10

        grads = tape.gradient(loss_total, model_dnn_pinn.trainable_weights)

        loss_list_pinn.append(float(loss))
        loss_final = np.min(loss_list_pinn)
        optimizer.apply_gradients(zip(grads, model_dnn_pinn.trainable_weights))

        pred_out = model_dnn_pinn(inp_comb_test)
        test_loss_ini = pred_out - test_out
        test_loss = K.mean(K.square(test_loss_ini))
        test_loss_list_pinn.append(float(test_loss))

    # If the training loss reaches a minimum value of 0.01, or the maximum number of epochs is reached, the training process is stopped
        if (loss_final <= 0.01) | (epoch == epochs - 1):
            print("PINNfreq model1 training Completed. Epoch %d/%d -- loss: %.4f" % (epoch, epochs, float(loss)))
            break


    pred_out = model_dnn_pinn(inp_comb_test)
    corr_pinn = np.corrcoef(np.concatenate(test_out)[:], np.concatenate(pred_out)[:])[0][1]
    rmse_pinn = np.sqrt(np.mean(np.square(
        np.concatenate(scaler_out.inverse_transform(np.concatenate(test_out)[:][:, None])) -
        np.concatenate(scaler_out.inverse_transform(np.concatenate(pred_out)[:][:, None])))))
    print('#### PINNfreq Performance ####')
    print('Corr: %.2f,  RMSE: %.1f'%(corr_pinn, rmse_pinn))


    model_save_path = "./saved/model/PINNfreq.h5"  # HDF5格式
    model_dnn_pinn.save(model_save_path)
    print(f"模型已保存至：{model_save_path}")
# 训练结束后保存损失
    save_losses(loss_list_pinn, test_loss_list_pinn)

    return model_dnn_pinn


