import numpy as np

from Constant import *
from Model_Structure import *
from dataset import *


if __name__ == "__main__":
    batch_size = 8
    epochs = 10
    # 加载模型
    model = radio2016_model(lr=1e-4)
    # 加载数据
    X_train, Y_train, X_test, Y_test = load_radio2016_regress(mod_snr=('8PSK', 18))

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=1e-5)
    model_checkpoint = ModelCheckpoint("Weights/signal.h5", monitor="val_loss",
                                       save_best_only=True, save_weights_only=False, verbose=1)
    callbacks = [model_checkpoint, lr_reducer]
    model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
              callbacks=callbacks, validation_data=(X_test, Y_test), verbose=1)
    try:
        model.load_weights("Weights/signal.h5")
    except:
        print("重载失败")
    print(model.predict(np.array([X_test[0]])))
    print(Y_test[0])

    print("end")
