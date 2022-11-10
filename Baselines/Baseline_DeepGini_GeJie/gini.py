import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   # 当前工程路径

import numpy as np


def Gini_score(confidence):
    confidence = np.array(confidence)
    gini = np.sum(confidence * confidence, axis=1)
    return 1-gini


if __name__ == "__main__":
    print("gini")



