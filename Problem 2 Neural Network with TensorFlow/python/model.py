from tensorflow.keras import Model
import keras


# The decorator "@keras.saving.register_keras_serializable()" allows you to save
# your trained model to a .keras file and then load it from the file for testing.
@keras.saving.register_keras_serializable()
class MyModel(Model):
    """Neural network to classify MNIST images."""

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        # TODO: Define the layers of your neural network

    def call(self, x, training=False):
        """
        Forward pass.

        Parameters
        ----------
        x : tensor float32 (28, 28)
            Input MNIST image.
        training : bool, optional
            training=True is only needed if there are layers with different
            behavior during training versus inference (e.g. Dropout).
            The default is False.

        Returns
        -------
        out : tensor float32 (None, 10)
              Class probabilities.

        """

        # TODO: Implement the forward pass through the layers of your neural network
        pass
