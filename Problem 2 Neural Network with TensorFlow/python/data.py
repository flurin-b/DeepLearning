import numpy as np
import tensorflow as tf


class DataLoaderMNIST:
    """Provide train, validation, and test datasets of the MNIST dataset."""

    def __init__(self, validation_dataset_size=5000, mini_batch_size=32):
        # Load MNIST data
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # TODO
        # - Preprocess images
        #   - Convert images to float32
        #   - Normalize images to values between 0 and 1
        # - Split validation dataset from training dataset
        # - Convert labels to one-hot tensors

        # Create datasets
        self._train_dataset = ...  # Use batching and shuffling
        self._valid_dataset = ...  # Use batching
        self._test_dataset = ...  # Use batching

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset
