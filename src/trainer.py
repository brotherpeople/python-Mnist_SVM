import logging
import wandb
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class ModelTrainer:
    """
    Handles the training process for both neural network and SVM models.

    This class implements training procedures with proper logging,
    checkpointing, and early stopping capabilities.

    Attributes:
        config (dict): Training configuration parameters
        logger (logging.Logger): Logger instance for training progress
    """

    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)

    def train_neural_network(self, model, x_train, y_train, x_val, y_val):
        """
        Trains the neural network model with the provided data.

        Implements training best practices including:
        - Model checkpointing
        - Early stopping
        - Progress logging
        - Validation monitoring

        Args:
            model (tf.keras.Model): The neural network model to train
            x_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            x_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels

        Returns:
            tf.keras.callbacks.History: Training history
        """

        callbacks = [
            ModelCheckpoint(
                "best_model.keras", save_best_only=True, monitor="val_accuracy"
            ),
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ]

        history = model.fit(
            x_train,
            y_train,
            batch_size=self.config["training"]["batch_size"],
            epochs=self.config["training"]["epochs"],
            validation_data=(x_val, y_val),
            callbacks=callbacks,
        )

        self.logger.info("Neural Network training completed")
        return history

    def train_svm(self, model, x_train, y_train):
        model.fit(x_train.reshape(len(x_train), -1), y_train)
        self.logger.info("SVM training completed")
