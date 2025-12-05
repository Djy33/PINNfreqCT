from model.CNN import *

def save_losses(loss_list, test_loss_list, save_path="./saved/loss/losses_PINN.pkl"):
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
def load_losses(load_path="./saved/loss/losses_PINN.pkl"):
    global loss_list_pinn, test_loss_list_pinn
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    loss_list_pinn = data['loss_list_pinn']
    test_loss_list_pinn = data['test_loss_list_pinn']

# Two lists are initialized to keep track of the training and testing loss during each epoch
loss_list_pinn = []
test_loss_list_pinn = []

def pinn():
    model_dnn_pinn = model_DNN(np.shape(train_beats)[-1], 1, 64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)
    epochs = 5000
    for epoch in range(epochs):
        with tf.GradientTape() as tape:

            tape.watch(inp_comb)
        # Traditional out
            yh = model_dnn_pinn(inp_comb, training=True)
            loss_ini = yh - train_out
            loss = K.mean(K.square(loss_ini))

        # Additional tf.GradientTape contexts are used to compute the derivatives of the model's predictions with respect to the features
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

        # A physics-based prediction is computed by adding the model's predictions to the product of the computed derivatives and
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
            print("PINN model training Completed. Epoch %d/%d -- loss: %.4f" % (epoch, epochs, float(loss)))
            break
    pred_out = model_dnn_pinn(inp_comb_test)
    corr_pinn = np.corrcoef(np.concatenate(test_out)[:], np.concatenate(pred_out)[:])[0][1]
    rmse_pinn = np.sqrt(np.mean(np.square(
        np.concatenate(scaler_out.inverse_transform(np.concatenate(test_out)[:][:, None])) -
        np.concatenate(scaler_out.inverse_transform(np.concatenate(pred_out)[:][:, None])))))
    print('#### PINN Performance ####')
    print('Corr: %.2f,  RMSE: %.1f'%(corr_pinn, rmse_pinn))


    model_save_path = "./saved/model/PINN.h5"  # HDF5格式
    model_dnn_pinn.save(model_save_path)
    print(f"模型已保存至：{model_save_path}")
# 训练结束后保存损失
    save_losses(loss_list_pinn, test_loss_list_pinn)

    return model_dnn_pinn