import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径
import numpy as np
from common import *
from keras.backend import function


def neuro_freq_rank(model, trainset, testset, ideal_rank):
    X_train = trainset[0]   # 图像
    Y_train = trainset[1]   # 类标
    bug_train = trainset[2] # bug标
    X_test = testset[0]
    Y_test = testset[1]
    bug_test = testset[2]

    neuro_positive, neuro_negtive = neuro_freq_train(model, X_train, Y_train)  # 对训练集的神经元激活
    x, y, bug, select_ls = neuro_rank_test(model, X_test, Y_test, bug_test, neuro_positive, neuro_negtive)

    # 计算指标
    apfd_all = improved_APFD(ideal_rank, bug)
    print("APFD分数：%.3f%%" % (apfd_all * 100))
    return select_ls, (x, y, bug)


def neuro_freq_train(model, X, Y):
    # 计算训练集各类样本的神经元激活频率
    layer_dense = function(inputs=model.inputs, outputs=[model.layers[-2].output])
    neuro_num = layer_dense.outputs[0].shape.dims[1].value     # 神经元数
    act = layer_dense([X])[0]   # 统计倒数第二层神经元激活
    act = np.where(act > 0, 1, 0)

    neuro_act = []
    for cla in range(np.max(Y)+1):
        index = np.where(Y == cla)
        neuro_act.append(np.sum(act[index], axis=0))
    neuro_act = np.array(neuro_act)

    # 对神经元排序
    neuro_freq = np.argsort(-neuro_act, axis=1)
    neuro_positive = neuro_freq[..., :int(neuro_num*0.5)]
    neuro_negtive = neuro_freq[..., int(neuro_num*0.5):]
    return neuro_positive, neuro_negtive


def neuro_rank_test(model, X, Y_true, bug, neuro_pos, neuro_neg):
    # 计算测试样本在已有频率里的得分
    pred = model.predict(X)
    Y = np.argmax(pred, axis=1)     # 获得模型对样本的预测类标

    # 获得测试样本的激活频率
    layer_dense = function(inputs=model.inputs, outputs=[model.layers[-2].output])
    neuro_num = layer_dense.outputs[0].shape.dims[1].value     # 神经元数
    test_act = layer_dense([X])[0]   # 统计倒数第二层神经元激活
    test_act = np.where(test_act > 0, 1, 0) # 设置神经元阈值
    test_neuro_pos = np.argsort(-test_act, axis=1)[..., :int(neuro_num * 0.2)]
    test_neuro_neg = np.argsort(-test_act, axis=1)[..., int(neuro_num * 0.2):]

    score = []  # 存放每个样本的分数
    for idx in range(len(X)):
        label = Y[idx]  # 样本类标
        train_neuro_pos = neuro_pos[label]  # 对应训练集的神经元激活顺序
        train_neuro_neg = neuro_neg[label]
        test_pos = test_neuro_pos[idx]  # 样本实际的神经元激活顺序
        test_neg = test_neuro_neg[idx]
        score_pos = np.size(np.intersect1d(train_neuro_pos, test_pos))/np.size(train_neuro_pos)
        score_neg = np.size(np.intersect1d(train_neuro_neg, test_neg))/np.size(train_neuro_neg)

        score.append(score_pos/(score_pos+score_neg))  # 交集数
    score = np.array(score)
    select_ls = np.argsort(1-score)
    return X[select_ls], Y_true[select_ls], bug[select_ls], select_ls


if __name__ == "__main__":
    from Model.Model_Lenet5 import *
    from Load_dataset import *
    from common import *
    from Draw import *

    base_path = os.path.dirname(__file__)
    # 加载模型
    # model_path = os.path.join(base_path, "../../Model_weight/Cifar10/Lenet5/Wed-Feb-23-20:33:29-2022.h5")
    model_path = os.path.join(base_path, "../../Model_weight/Mnist/Lenet5/Sat-Jun-18-21:52:57-2022.h5")
    # model = lenet5_cifar10(path=model_path)
    model = lenet5_mnist(path=model_path)

    # 加载数据
    # data_path = os.path.join(base_path, "../../Adv/Cifar10/Lenet5/CW_perclass=400.npz")
    data_path = os.path.join(base_path, "../../Adv/Mnist/Lenet5/FGSM_perclass=900.npz")
    X_train_clean, X_train_adv, Class_train, X_test_clean, X_test_adv, Class_test = load_clean_adv(data_path)

    # _, _, _, _, X_test_fp, Y_test_fp = load_cifar10(onehot=False, sequence=False, split=False)
    # _, _, X_test_adv, _ = filter_sample(model, X_test_fp, Y_test_fp)

    # train_adv_num = 500
    # train_clean_num = 500
    test_adv_num = 1000
    test_clean_num = 4000

    X_clean_test = X_test_clean[:test_clean_num]
    X_adv_test = X_test_adv[:test_adv_num]
    X_test = np.concatenate((X_adv_test, X_clean_test), axis=0)
    Class_test = np.concatenate((Class_test[:test_adv_num], Class_test[:test_clean_num]), axis=0)
    Y_test = np.array([1]*test_adv_num + [0]*test_clean_num)

    neuro_positive, neuro_negtive = neuro_freq_train(model, X_train_clean, Class_train)    # 对训练集的神经元激活
    _, _, rank = neuro_rank_test(model, X_test, Class_test, neuro_positive, neuro_negtive)

    ideal = Y_test
    real = Y_test[rank]
    apfd = improved_APFD(ideal, real)
    print("APFD分数：%.5f" % apfd)

    plot_data(path="neuro.png", data1=curve(ideal), label1="Ideal",
              data2=curve(real), label2="Neuro",
              Xlabel="Sample", Ylabel="The cumulative number of errors", show=False)
    print("end")


