import os

SEED = 0

RML2016_FEATURE_DIM = (2, 127)  # 数据特征维度
RML2016_CLASS_NUM = 2     # 两路回归预测

GPU = 4
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)


if __name__ == "__main__":
    print("end")
