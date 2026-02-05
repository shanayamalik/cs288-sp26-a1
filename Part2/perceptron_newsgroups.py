"""Perceptron model for 20 Newsgroups dataset.

This script trains a perceptron on 20 Newsgroups and generates test predictions.
Runs with no command-line arguments.
"""

import os
from perceptron import PerceptronModel, featurize_data
from utils import DataType, load_data


def main():
    # hyperparameters  
    data_type = DataType.NEWSGROUPS
    feature_types = {"bow"}  # bag-of-words baseline
    num_epochs = 3
    learning_rate = 0.1
    
    print("Loading 20 Newsgroups data...")
    train_data, val_data, dev_data, test_data = load_data(data_type)
    
    print("Featurizing data...")
    train_data = featurize_data(train_data, feature_types)
    val_data = featurize_data(val_data, feature_types)
    dev_data = featurize_data(dev_data, feature_types)
    test_data = featurize_data(test_data, feature_types)
    
    print("Training perceptron model...")
    model = PerceptronModel()
    model.train(train_data, val_data, num_epochs, learning_rate)
    
    # evaluate on dev set
    dev_acc = model.evaluate(
        dev_data,
        save_path="results/perceptron_newsgroups_dev_predictions.csv"
    )
    print(f"Development accuracy: {100 * dev_acc:.2f}%")
    
    # generate test predictions 
    model.evaluate(
        test_data,
        save_path="results/perceptron_newsgroups_test_predictions.csv"
    )
    print("Test predictions saved to results/perceptron_newsgroups_test_predictions.csv")
    
    # save model weights
    model.save_weights("results/perceptron_newsgroups_model.json")


if __name__ == "__main__":
    main()