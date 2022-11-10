import os
import numpy as np
import re
from Constant import *
from tqdm import tqdm


def find_files(file, reg):
    # 遍历file下的所有满足正则化条件的文件（.wav）
    files_list = list()  # 文件的绝对路径
    pattern = re.compile(reg)
    for root, dirs, files in os.walk(file):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示文件夹路径
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            if pattern.search(f):   # 正则化匹配模板
                files_list.append(os.path.join(root, f))
            # if f.split(".")[-1] in types:
            #     files_list.append(os.path.join(root, f))
    return files_list


def find_member(list, reg):
    # 遍历list下所有满足正则化条件的成员
    men_list = []
    pattern = re.compile(reg)
    for i in list:
        if pattern.search(i):
            men_list.append(i)
    return men_list


def collect_data(path, class_num=None):
    # 将路径下的文件进行分类
    # class_num 每类的数量
    # path = "Dataset/VCTK/NPZ/test"
    test_path = sorted(find_files(path, ".npz"))
    test_name = []
    for i in test_path:
        test_name.append(i.split("/")[-1].split("-")[0])    # 人名
    test_people = sorted(list(set(test_name)))

    X_test = []  # 文件路径
    Y_test = []  # 文件类标
    N_test = []  # 名字
    for idx, i in enumerate(test_people):
        if type(class_num) != type(None):
            x = find_member(test_path, i)[:class_num]   # 每类是否做数量限制
        else:
            x = find_member(test_path, i)
        y = [idx] * len(x)
        n = [i] * len(x)
        X_test.extend(x)
        Y_test.extend(y)
        N_test.extend(n)
    return X_test, Y_test, N_test


def creat_dataset():
    # 创建数据集
    X_test, Y_test, N_test = collect_data("Dataset/VCTK/NPZ/test", class_num=40)    # 测试集每类40个
    X_train, Y_train, N_train = collect_data("Dataset/VCTK/NPZ/train", class_num=250)   # 训练集每类250个
    np.savez("Dataset/VCTK/Vctk_dataset.npz", X_test=X_test, Y_test=Y_test, N_test=N_test,
             X_train=X_train, Y_train=Y_train, N_train=N_train)     # 对这个npz数据索引即可


def load_coded_sp(path):
    feature = np.load(path)
    coded_sp = feature["coded_sp"]

    return coded_sp


def load_all_feature(path):
    feature = np.load(path)
    f0 = feature["f0"]
    coded_sp = feature["coded_sp"]
    ap = feature["ap"]
    fs = feature["fs"]
    return f0, coded_sp, ap, fs


def load_dataset():
    vctk = np.load("Dataset/VCTK/Vctk_dataset.npz")
    X_train, Y_train, N_train, X_test, Y_test, N_test = vctk["X_train"], vctk["Y_train"], vctk["N_train"], \
                                                        vctk["X_test"], vctk["Y_test"], vctk["N_test"]
    X_train = X_train[:VCTK_CLASS_NUM * VCTK_TRAIN_NUM]  # 取前VCTK_CLASS_NUM类的样本
    Y_train = Y_train[:VCTK_CLASS_NUM * VCTK_TRAIN_NUM]
    N_train = N_train[:VCTK_CLASS_NUM * VCTK_TRAIN_NUM]
    X_test = X_test[:VCTK_CLASS_NUM * VCTC_TEST_NUM]
    Y_test = Y_test[:VCTK_CLASS_NUM * VCTC_TEST_NUM]
    N_test = N_test[:VCTK_CLASS_NUM * VCTC_TEST_NUM]

    X_train = np.array([load_coded_sp(i) for i in tqdm(X_train)]).reshape(-1, 601, 64)
    X_test = np.array([load_coded_sp(i) for i in tqdm(X_test)]).reshape(-1, 601, 64)

    return X_train, Y_train, N_train, X_test, Y_test, N_test


if __name__ == "__main__":
    # data = np.load("/public/liujiawei/ZHB/GeJie/Speech/Dataset/VCTK/NPZ/test/p226-006.npz")
    # f0 = data["f0"]
    # coded_sp = data["coded_sp"]
    # ap = data["ap"]
    # fs = data["fs"]

    # creat_dataset()

    X_train, Y_train, N_train, X_test, Y_test, N_test = load_dataset()

    print("end")





