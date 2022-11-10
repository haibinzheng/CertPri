from Constant import *
import numpy as np


def normalization(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))


def load_radio2016(mod_snr=('8PSK', 18)):
    # 加载特定类型的数据
    radio_2016a = np.load("Dataset/RML2016A/RML2016.10a_Norm.npz")
    X_train = radio_2016a["X_train"]
    Y_train = radio_2016a["Y_train"]
    Mod_snr_train = radio_2016a["Mod_snr_train"]
    X_test = radio_2016a["X_test"]
    Y_test = radio_2016a["Y_test"]
    Mod_snr_test = radio_2016a["Mod_snr_test"]

    # 取出满足调制类型和信噪比条件的数据索引
    mod_snr = np.array(mod_snr)
    train_idx = [i for i in range(len(Mod_snr_train)) if all(Mod_snr_train[i] == mod_snr)]
    test_idx = [i for i in range(len(Mod_snr_test)) if all(Mod_snr_test[i] == mod_snr)]

    # 取出数据
    X_train = X_train[train_idx]
    Y_train = Y_train[train_idx]
    X_test = X_test[test_idx]
    Y_test = Y_test[test_idx]

    return X_train, Y_train, X_test, Y_test


def load_radio2016_regress(mod_snr=('8PSK', 18)):
    # 加载回归数据集
    X_train, _, X_test, _ = load_radio2016(mod_snr=mod_snr)
    X_train = normalization(X_train)
    X_test = normalization(X_test)

    # 将数据集做成回归数据集
    Y_train = X_train[..., -1]
    X_train = X_train[..., :-1]

    Y_test = X_test[..., -1]
    X_test = X_test[..., :-1]
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_radio2016_regress(mod_snr=('QPSK', 18))

    print("end")
