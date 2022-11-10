from Constant import *
from Model_Structure import *
from dataset import *


if __name__ == "__main__":
    batch_size = 64
    epochs = 10
    # 加载模型
    model = vctk_model()
    # 加载数据
    X_train, Y_train, N_train, X_test, Y_test, N_test = load_dataset()

    model_checkpoint = ModelCheckpoint("Weights/speech.h5", monitor="val_loss",
                                       save_best_only=True, save_weights_only=False, verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=1e-5)
    callbacks = [lr_reducer, model_checkpoint]
    model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
              callbacks=callbacks, validation_data=(X_test, Y_test), verbose=1)
    print("end")


