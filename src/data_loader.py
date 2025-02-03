import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


class MNISTDataLoader:
    """
    A class responsible for loading and preprocessing the MNIST dataset.

    This class handles data loading, normalization, and train/validation/test splitting.
    It implements best practices for data preprocessing in machine learning pipelines.

    Attributes:
        config (dict): Configuration dictionary containing data parameters
        x_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        x_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation labels
        x_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
    """

    def __init__(self, config):
        self.config = config
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def load_data(self):
        """
        Loads and preprocesses the MNIST dataset.

        The preprocessing steps include:
        1. Loading raw data from TensorFlow datasets
        2. Normalizing pixel values to [0,1] range
        3. Splitting training data into train and validation sets

        Returns:
            tuple: (x_train, y_train, x_val, y_val, x_test, y_test)
        """

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train,
            y_train,
            test_size=self.config["training"]["validation_split"],
            random_state=42,
        )

        self.x_test = x_test
        self.y_test = y_test

        return (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
        )
