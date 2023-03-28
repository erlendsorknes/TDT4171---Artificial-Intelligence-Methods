import pickle
from typing import Dict, List, Any, Union
import numpy as np
# Keras
import tensorflow as tf
from tensorflow import keras

from keras.utils import pad_sequences
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set the GPU device to use


def load_data() -> Dict[str, Union[List[Any], int]]:
    path = "keras-data.pickle"
    with open(file=path, mode="rb") as file:
        data = pickle.load(file)

    return data


def preprocess_data(data: Dict[str, Union[List[Any], int]]) -> Dict[str, Union[List[Any], np.ndarray, int]]:
    """
    Preprocesses the data dictionary. Both the training-data and the test-data must be padded
    to the same length; play around with the maxlen parameter to trade off speed and accuracy.
    """
    maxlen = data["max_length"] // 16
    data["x_train"] = pad_sequences(data['x_train'], maxlen=maxlen)
    data["y_train"] = np.asarray(data['y_train'])
    data["x_test"] = pad_sequences(data['x_test'], maxlen=maxlen)
    data["y_test"] = np.asarray(data['y_test'])
    data['vocab_size'] = np.asarray(data['vocab_size'])

    return data


def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward") -> float:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The accuracy of the model on test data
    """



    ## Creating a simple feedforward recurrent network
    if (model_type == 'feedforward'):
        model = tf._keras.Sequential()
        model.add(tf.keras.layers.Embedding(data['vocab_size'], 16, input_length=data["x_train"].shape[1]))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(data["x_train"], data["y_train"], epochs=5, batch_size=500,
                  validation_data=(data["x_train"], data["y_train"]))

        _, accuracy = model.evaluate(data["x_test"], data["y_test"], verbose=2)

        return accuracy

    ## Creating a simple recurrent network
    if (model_type == 'reccurent' ):
        model = tf._keras.Sequential()
        model.add(tf.keras.layers.Embedding(data['vocab_size'], 16, input_length=data["x_train"].shape[1]))
        model.add(tf.keras.layers.LSTM(16))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy"])

        model.fit(data["x_train"], data["y_train"], epochs=3, batch_size=500,
                  validation_data=(data["x_train"], data["y_train"]))
        _, accuracy = model.evaluate(data["x_test"], data["y_test"], verbose=2)

        return accuracy


def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    fnn_test_accuracy = train_model(keras_data, model_type="feedforward")
    print('Model: feedforward NN.\n'
          f'Test accuracy: {fnn_test_accuracy:.3f}')
    print("4. Training reccurent neural network...")
    fnn_test_accuracy = train_model(keras_data, model_type="reccurent")
    print('Model: reccurent NN.\n'
          f'Test accuracy: {fnn_test_accuracy:.3f}')


if __name__ == '__main__':
    main()
