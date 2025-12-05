from model.PINNfreq import *

# 在训练完成后保存损失列表
def save_losses(loss_list, test_loss_list, save_path="./saved/loss/losses_CNNfreq.pkl"):
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
def load_losses(load_path="./saved/loss/losses_CNNfreq.pkl"):
    global loss_list_conv, test_loss_list_conv
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    loss_list_conv = data['loss_list_conv']
    test_loss_list_conv = data['test_loss_list_conv']

# Two lists are initialized to keep track of the training and testing loss during each epoch
loss_list_conv = []
test_loss_list_conv = []

def cnn_freq():
    #############################
    ###### Conventional model1
    #############################

    # A Deep Neural Network model1 is initialized with the dimension of the beats, the diemnsion of each feature, and the number of neurons in the first dense layer(CNN提取特征数量)
    T = train_beats.shape[1]
    F = train_beats.shape[2]
    model_dnn_conv = model_DNN(T, F, 1, 64)
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
            print("CNNfreq model1 training Completed. Epoch %d/%d -- loss: %.4f" % (epoch, epochs, float(loss)))
            break
    # The trained model1's predictions on the test dataset are computed
    pred_out = model_dnn_conv(inp_comb_test)

    # The Pearson correlation coefficient and the Root Mean Square Error are calculated between the actual and predicted test outcomes
    corr_conv = np.corrcoef(np.concatenate(test_out)[:], np.concatenate(pred_out)[:])[0][1]
    rmse_conv = np.sqrt(np.mean(np.square
                                (np.concatenate(scaler_out.inverse_transform(np.concatenate(test_out)[:][:, None])) -
                                 np.concatenate(scaler_out.inverse_transform(np.concatenate(pred_out)[:][:, None])))))
    print('#### CNNfreq Performance ####')
    print('Corr: %.2f,  RMSE: %.1f' % (corr_conv, rmse_conv))

    model_save_path = "./saved/model/CNNfreq.h5"  # HDF5格式
    model_dnn_conv.save(model_save_path)
    print(f"模型已保存至：{model_save_path}")

    # 训练结束后保存损失
    save_losses(loss_list_conv, test_loss_list_conv)
    return model_dnn_conv