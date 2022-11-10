import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径

import numpy as np
from common import *


def srs_rank(testset, ideal_rank):
    X_test = testset[0]     # 图像
    Y_test = testset[1]     # 类标
    bug_test = testset[2]   # bug标

    length = len(X_test)     # 样本数
    select_ls = np.random.choice(list(range(length)), size=length, replace=False)
    x = X_test[select_ls]
    y = Y_test[select_ls]
    bug = bug_test[select_ls]

    # 进一步计算指标
    apfd_all = RAUC(ideal_rank, bug)
    print("APFD分数：%.3f%%" % (apfd_all * 100))
    return select_ls, (x, y, bug)


if __name__ == "__main__":

    print("end")

