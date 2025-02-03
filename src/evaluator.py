import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import logging


class ModelEvaluator:
    """
    Handles model evaluation and performance metric calculation.

    This class provides comprehensive evaluation capabilities including
    accuracy metrics, confusion matrices, and classification reports.

    Attributes:
        logger (logging.Logger): Logger instance for evaluation results
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_neural_network(self, model, x_test, y_test):
        """
        Evaluates the neural network model performance.

        Calculates and logs various performance metrics including:
        - Test accuracy
        - Confusion matrix
        - Precision, recall, and F1-score

        Args:
            model (tf.keras.Model): Trained neural network model
            x_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels

        Returns:
            tuple: (test_loss, test_accuracy)
        """
        test_loss, test_acc = model.evaluate(x_test, y_test)
        y_pred = np.argmax(model.predict(x_test), axis=1)

        self._print_metrics(y_test, y_pred)
        return test_loss, test_acc

    def evaluate_svm(self, model, x_test, y_test):
        x_test_reshaped = x_test.reshape(len(x_test), -1)
        y_pred = model.predict(x_test_reshaped)

        self._print_metrics(y_test, y_pred)
        return model.score(x_test_reshaped, y_test)

    def _print_metrics(self, y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        self.logger.info(f"Confusion Matrix:\n{conf_matrix}")
        self.logger.info(f"Classification Report:\n{report}")
