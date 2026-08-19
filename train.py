"""
SageMaker script-mode training entry point for the Fashion-MNIST CNN.

SageMaker copies this file into the training container and calls it as:
    python train.py --epochs 10 --batch-size 128 --learning-rate 0.001
Data channels and the model output directory are passed as environment
variables (SM_CHANNEL_TRAIN, SM_CHANNEL_TEST, SM_MODEL_DIR) by the
TensorFlow estimator configured in the notebook.
"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(kernel_size=(3, 3), filters=(32, 64)):
    inputs = layers.Input(shape=(28, 28, 1), name="input")
    x = layers.Conv2D(filters[0], kernel_size, strides=1, padding="same",
                       activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters[1], kernel_size, strides=1, padding="same",
                       activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="cnn_fashion_mnist")
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "."))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "."))
    return parser.parse_args()


def load_data(train_dir, test_dir):
    x_train = np.load(os.path.join(train_dir, "x_train.npy"))
    y_train = np.load(os.path.join(train_dir, "y_train.npy"))
    x_test = np.load(os.path.join(test_dir, "x_test.npy"))
    y_test = np.load(os.path.join(test_dir, "y_test.npy"))
    return x_train, y_train, x_test, y_test


def main():
    args = parse_args()
    x_train, y_train, x_test, y_test = load_data(args.train, args.test)

    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # SageMaker's TensorFlow serving container expects a SavedModel under
    # <model_dir>/<version>/
    export_path = os.path.join(args.model_dir, "1")
    model.export(export_path)


if __name__ == "__main__":
    main()
