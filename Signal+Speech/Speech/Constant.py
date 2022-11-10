import os

SEED = 0

VCTK_FEATURE_DIM = (601, 64)  # 数据特征维度
VCTK_CLASS_NUM = 10     # 总共30类，取前10类
VCTK_TRAIN_NUM = 250
VCTC_TEST_NUM = 40

GPU = 4
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)


if __name__ == "__main__":
    print("end")

