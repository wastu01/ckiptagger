import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print(tf.test.gpu_device_name())

# 載入數據
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# 正規化輸入數據
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 使用更小的自定義模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(100, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
history = model.fit(
    x_train[:1000],  # 只使用前1000個樣本
    y_train[:1000],
    epochs=2,  # 減少訓練輪數
    batch_size=32,  # 批次大小
    validation_split=0.2,  # 使用20%的訓練數據作為驗證集
    verbose=1  # 顯示訓練進度
)

# 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')