import tensorflow as tf
from sklearn.svm import LinearSVC


class NeuralNetwork:
    """
    Neural Network model for MNIST digit classification.

    This class implements a simple but effective neural network architecture
    using TensorFlow. The network consists of fully connected layers with
    dropout for regularization.

    Attributes:
        config (dict): Model configuration parameters
        model (tf.keras.Model): The compiled Keras model
    """

    def __init__(self, config):
        self.config = config
        self.model = self._build_model()

    def _build_model(self):
        """
        Constructs the neural network architecture.

        The architecture consists of:
        1. Flatten layer to convert 2D images to 1D vectors
        2. Dense hidden layer with ReLU activation
        3. Dropout layer for regularization
        4. Output layer with softmax activation

        Returns:
            tf.keras.Model: Compiled Keras model
        """

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(
                    input_shape=(
                        self.config["data"]["img_size"],
                        self.config["data"]["img_size"],
                    )
                ),
                tf.keras.layers.Dense(
                    self.config["model"]["dense_units"], activation="relu"
                ),
                tf.keras.layers.Dropout(self.config["model"]["dropout_rate"]),
                tf.keras.layers.Dense(
                    self.config["data"]["num_classes"], activation="softmax"
                ),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config["training"]["learning_rate"]
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model


class SVMClassifier:
    """MNIST 분류를 위한 SVM 모델 클래스입니다."""

    def __init__(self):
        self.model = LinearSVC(random_state=42)
