"""Batching benchmark script for MLP.

Measures wall-time speed (seconds per 1000 examples) for different batch sizes.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multilayer_perceptron import (
    Tokenizer,
    BOWDataset,
    MultilayerPerceptronModel,
    get_label_mappings,
)
from utils import DataType, load_data


def benchmark_batch_size(model, dataset, batch_size, device, num_runs=5):
    """Benchmark inference speed for a given batch size.
    
    Args:
        model: The MLP model
        dataset: The dataset to use
        batch_size: Batch size to test
        device: 'cuda' or 'cpu'
        num_runs: Number of runs to average over
        
    Returns:
        (mean_time, std_time) in seconds per 1000 examples
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Limit to ~1000 examples
    num_examples = min(1000, len(dataset))
    num_batches = (num_examples + batch_size - 1) // batch_size
    
    times = []
    
    for run in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            for i, (inputs_b_l, lengths_b, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                inputs_b_l = inputs_b_l.to(device)
                lengths_b = lengths_b.to(device)
                
                # Forward pass
                outputs = model(inputs_b_l, lengths_b)
                
                # Ensure computation completes (synchronize GPU)
                if device == 'cuda':
                    torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        
        # Normalize to time per 1000 examples
        time_per_1000 = (elapsed / num_examples) * 1000
        times.append(time_per_1000)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time


def main():
    print("=" * 80)
    print("MLP BATCHING BENCHMARK")
    print("=" * 80)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cpu':
        print("WARNING: Running on CPU. Results will be slow. Enable GPU in Colab!")
    print()
    
    # Load data (using SST-2 for benchmark)
    print("Loading SST-2 data...")
    data_type = DataType.SST2
    train_data, val_data, dev_data, test_data = load_data(data_type)
    
    # Create tokenizer and dataset
    print("Creating tokenizer and dataset...")
    tokenizer = Tokenizer(train_data, max_vocab_size=20000)
    label2id, id2label = get_label_mappings(train_data)
    
    # Use dev data for benchmarking (smaller, faster)
    benchmark_dataset = BOWDataset(dev_data, tokenizer, label2id, max_length=100)
    print(f"Dataset size: {len(benchmark_dataset)} examples")
    
    # Create model
    print("Creating model...")
    model = MultilayerPerceptronModel(
        vocab_size=len(tokenizer.token2id),
        num_classes=len(label2id),
        padding_index=tokenizer.TOK_PADDING_INDEX,
        activation="relu",  # Using ReLU for benchmarking
    )
    model = model.to(device)
    model.eval()
    
    # Batch sizes to test
    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Batch Size':<15} {'Mean Time (s)':<20} {'Std Dev (s)':<20} {'Speedup':<10}")
    print("-" * 80)
    
    results = []
    baseline_time = None
    
    for batch_size in batch_sizes:
        print(f"Testing batch_size={batch_size}...", end=" ", flush=True)
        
        mean_time, std_time = benchmark_batch_size(
            model, benchmark_dataset, batch_size, device, num_runs=5
        )
        
        # Calculate speedup relative to batch_size=1
        if baseline_time is None:
            baseline_time = mean_time
            speedup = 1.0
        else:
            speedup = baseline_time / mean_time
        
        results.append({
            'batch_size': batch_size,
            'mean_time': mean_time,
            'std_time': std_time,
            'speedup': speedup
        })
        
        print(f"{batch_size:<15} {mean_time:<20.4f} {std_time:<20.4f} {speedup:<10.2f}x")
    
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Processing time is for 1,000 examples")
    print(f"  - Each batch size tested 5 times")
    print(f"  - Speedup is relative to batch_size=1")
    print(f"  - Device: {device.upper()}")
    
    if device == 'cuda':
        print(f"\n✅ GPU benchmarking complete!")
    else:
        print(f"\n⚠️  CPU benchmarking complete. Re-run on GPU for final results.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
