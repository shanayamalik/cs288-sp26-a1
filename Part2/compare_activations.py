"""Experiment script to compare different activation functions.

Compares ReLU, Sigmoid, and Tanh activation functions on SST-2 dataset.
"""

import torch
from multilayer_perceptron import (
    Tokenizer,
    BOWDataset,
    MultilayerPerceptronModel,
    Trainer,
    get_label_mappings,
)
from utils import DataType, load_data


def train_and_evaluate(activation: str, num_epochs: int = 3, lr: float = 0.001):
    """Train and evaluate a model with a specific activation function.
    
    Args:
        activation: Activation function ('relu', 'sigmoid', or 'tanh')
        num_epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        (train_acc, val_acc, dev_acc): Accuracy on each dataset
    """
    print(f"\n{'='*80}")
    print(f"Training with {activation.upper()} activation")
    print(f"{'='*80}")
    
    # Load data
    data_type = DataType.SST2
    train_data, val_data, dev_data, test_data = load_data(data_type)
    
    # Create tokenizer and datasets
    tokenizer = Tokenizer(train_data, max_vocab_size=20000)
    label2id, id2label = get_label_mappings(train_data)
    
    max_length = 100
    train_ds = BOWDataset(train_data, tokenizer, label2id, max_length)
    val_ds = BOWDataset(val_data, tokenizer, label2id, max_length)
    dev_ds = BOWDataset(dev_data, tokenizer, label2id, max_length)
    
    # Create model with specified activation
    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=tokenizer.TOK_PADDING_INDEX,
        activation=activation,
    )
    
    # Train
    trainer = Trainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer.train(train_ds, val_ds, optimizer, num_epochs)
    
    # Evaluate
    train_acc = trainer.evaluate(train_ds)
    val_acc = trainer.evaluate(val_ds)
    dev_acc = trainer.evaluate(dev_ds)
    
    print(f"\n{activation.upper()} Results:")
    print(f"  Training accuracy:   {100 * train_acc:.2f}%")
    print(f"  Validation accuracy: {100 * val_acc:.2f}%")
    print(f"  Development accuracy: {100 * dev_acc:.2f}%")
    
    return train_acc, val_acc, dev_acc


def main():
    print("="*80)
    print("ACTIVATION FUNCTION COMPARISON")
    print("="*80)
    print("\nComparing ReLU, Sigmoid, and Tanh activation functions on SST-2 dataset")
    print("Each model trained for 3 epochs with learning rate 0.001")
    
    activation_functions = ["relu", "sigmoid", "tanh"]
    results = {}
    
    for activation in activation_functions:
        train_acc, val_acc, dev_acc = train_and_evaluate(activation, num_epochs=3, lr=0.001)
        results[activation] = {
            'train': train_acc,
            'val': val_acc,
            'dev': dev_acc
        }
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Activation':<15} {'Train Acc':<15} {'Val Acc':<15} {'Dev Acc':<15}")
    print("-"*80)
    
    for activation in activation_functions:
        r = results[activation]
        print(f"{activation.upper():<15} {100*r['train']:<15.2f} {100*r['val']:<15.2f} {100*r['dev']:<15.2f}")
    
    # Find best activation
    best_activation = max(results.items(), key=lambda x: x[1]['dev'])
    print("="*80)
    print(f"\n✅ Best activation function: {best_activation[0].upper()}")
    print(f"   Development accuracy: {100 * best_activation[1]['dev']:.2f}%")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
