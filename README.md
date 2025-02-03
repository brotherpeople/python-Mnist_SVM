# MNIST Digit Classification Project

## Overview

This project demonstrates how to classify handwritten digits using two different machine learning approaches: Neural Networks and Support Vector Machines (SVM).

The project analyzes the famous MNIST dataset, which contains thousands of handwritten digits from 0 to 9. Think of it like teaching a computer to read handwritten numbers - something humans do easily but computers need careful training to accomplish.

## Project Structure

```
python-Mnist_SVM/
├── config/
│   └── config.yaml        # Settings for the models
├── src/
│   ├── data_loader.py     # Handles loading and preparing the data
│   ├── evaluator.py       # Checks how well the models perform
│   ├── models.py          # Contains the model designs
│   ├── trainer.py         # Code for training the models
│   └── visualizer.py      # Creates helpful visualizations
├── notebooks/
│   └── mnist_svm.ipynb    # Original experiment notebook
├── main.py                # Main program
└── requirements.txt       # Required packages
```

## Key Features

-   Loads and processes the MNIST dataset
-   Implements both neural network and SVM approaches
-   Shows how well each model performs
-   Creates visualizations to understand what's happening
-   Organized in a way that's easy to understand and modify

## Technical Details

The neural network model includes:

-   Input layer that takes in the digit images
-   Hidden layer with 512 neurons
-   Output layer that predicts which digit it sees

The SVM model:

-   Treats the problem as finding boundaries between different digits
-   Uses dimensionality reduction to help visualize the results
-   Shows how different digits are separated in 2D space

## How to Run the Project

### Setup

```bash
# Get the project files
git clone https://github.com/brotherpeople/python-Mnist_SVM.git
cd python-Mnist_SVM

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install needed packages
pip install -r requirements.txt
```

### Running the Program

```bash
python main.py
```

## Results and Visualization

One of the most interesting parts of this project is the visualization that shows how the SVM separates different digits. The colorful plot shows:

-   Where each digit typically appears in the reduced 2D space
-   How the model draws boundaries between different digits
-   Which digits are often confused with each other

## Learning Outcomes

Through this project, I gained practical experience with:

-   Working with real-world machine learning datasets
-   Implementing different classification approaches
-   Processing and visualizing high-dimensional data
-   Organizing a machine learning project properly
-   Using popular machine learning libraries like TensorFlow and scikit-learn

## Future Improvements

I plan to enhance this project by:

1. Adding more advanced neural network architectures
2. Implementing data augmentation techniques
3. Improving the visualization methods
4. Adding more detailed performance analysis

## License

This project is available under the MIT License.

---

The code and documentation structure greatly benefited from AI assistance, which helped ensure good coding practices while maintaining educational value. The original concept began with `notebooks/mnist_svm.ipynb` and was expanded into a full project with proper organization and additional features.
