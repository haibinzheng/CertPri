from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess=tf.compat.v1.Session(config=config)

# myInput = tf.placeholder(shape=(None,32,32,3), dtype=tf.float32)
# model = VGG19(input_tensor=myInput, classes=10, include_top=False, weights=None)
# del model
model = VGG19(input_shape=(32,32,3), classes=10, weights=None)
# model.summary()


MODEL_PATH = "/public/liujiawei/huawei/ZHB/ADF-master/models/"
model = load_model(MODEL_PATH+"cifar10_VGG19.h5")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=1, batch_size=512)

model.save(MODEL_PATH+"cifar10_VGG19.h5")

del model
import tensorflow as tf
sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)