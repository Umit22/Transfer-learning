import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import DenseNet121, VGG16, ResNet50
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Veri kümesini yükleme ve ön işleme
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)


# Model oluşturma
def create_model(model_type):
    if model_type == "DenseNet":
        base_model = DenseNet121(include_top=False, input_shape=(32, 32, 3))
    elif model_type == "VGGNet":
        base_model = VGG16(include_top=False, input_shape=(32, 32, 3))
    elif model_type == "ResNet":
        base_model = ResNet50(include_top=False, input_shape=(32, 32, 3))
    else:
        raise ValueError("Geçersiz model tipi!")

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    return model


# Modeli derleme ve eğitim
def train_model(model_type):
    model = create_model(model_type)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train_cat, epochs=10, batch_size=32, verbose=1)
    return model


# Modeli eğitme ve değerlendirme
def evaluate_model(model_type):
    model = train_model(model_type)
    y_pred = model.predict(x_test)
    y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
    acc = accuracy_score(y_test, y_pred_classes)
    print(f"{model_type} model doğruluğu: {acc}")


# Modelleri değerlendirme
evaluate_model("DenseNet")
evaluate_model("VGGNet")
evaluate_model("ResNet")
