import yaml
from src.data_loader import MNISTDataLoader
from src.models import NeuralNetwork, SVMClassifier
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.visualizer import ResultVisualizer


def main():
    # load configuration daya
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # load data
    data_loader = MNISTDataLoader(config)
    x_train, y_train, x_val, y_val, x_test, y_test = data_loader.load_data()

    # initialize model
    nn_model = NeuralNetwork(config)
    svm_model = SVMClassifier()

    # train
    trainer = ModelTrainer(config)
    nn_history = trainer.train_neural_network(
        nn_model.model, x_train, y_train, x_val, y_val
    )
    trainer.train_svm(svm_model.model, x_train, y_train)

    # evaluate
    evaluator = ModelEvaluator()
    nn_test_loss, nn_test_acc = evaluator.evaluate_neural_network(
        nn_model.model, x_test, y_test
    )
    svm_accuracy = evaluator.evaluate_svm(svm_model.model, x_test, y_test)

    # visualize
    visualizer = ResultVisualizer(config)
    visualizer.plot_training_history(nn_history)
    visualizer.visualize_svm_decision_boundary(svm_model.model, x_train, y_train)


if __name__ == "__main__":
    main()
