import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import numpy as np


class ResultVisualizer:
    """
    Handles visualization of training results and model performance.

    This class provides various visualization capabilities including
    learning curves and decision boundaries.

    Attributes:
        config (dict): Visualization configuration parameters
    """

    def __init__(self, config):
        self.config = config

    def plot_training_history(self, history):
        """
        Visualizes the training history of the neural network.

        Creates plots showing:
        - Training and validation accuracy over epochs
        - Training and validation loss over epochs

        Args:
            history (tf.keras.callbacks.History): Training history object
        """
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=self.config["visualization"]["fig_size"]
        )

        # accuracy graph
        ax1.plot(history.history["accuracy"], label="train")
        ax1.plot(history.history["val_accuracy"], label="validation")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        # loss graph
        ax2.plot(history.history["loss"], label="train")
        ax2.plot(history.history["val_loss"], label="validation")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def visualize_svm_decision_boundary(self, svm_model, x_train, y_train):
        """
        Visualizes the SVM decision boundary in 2D space with a color legend.

        Args:
            svm_model: Trained SVM model
            x_train: Training features
            y_train: Training labels
        """
        pca = PCA(n_components=self.config["visualization"]["pca_components"])
        x_train_2d = pca.fit_transform(x_train.reshape(len(x_train), -1))

        svm_2d = LinearSVC(random_state=42)
        svm_2d.fit(x_train_2d, y_train)

        # calculate decision boundary
        x_min, x_max = x_train_2d[:, 0].min() - 1, x_train_2d[:, 0].max() + 1
        y_min, y_max = x_train_2d[:, 1].min() - 1, x_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=self.config["visualization"]["fig_size"])

        contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap="viridis")

        for digit in range(10):
            mask = y_train == digit
            scatter = ax.scatter(
                x_train_2d[mask, 0],
                x_train_2d[mask, 1],
                label=f"Digit {digit}",
                s=20,
                alpha=0.6,
            )

        ax.set_title("SVM Decision Boundary (PCA 2D Projection)")
        ax.set_xlabel("First Principal Component")
        ax.set_ylabel("Second Principal Component")

        ax.legend(title="Digits", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()
