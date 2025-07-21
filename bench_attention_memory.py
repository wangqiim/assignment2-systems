import torch
import cs336_basics
import itertools
import tqdm
from torch.utils.data import Dataset, DataLoader

def benchmark_attention():
    batch_size = 8
    d_model_list = [16, 32, 64, 128]
    seq_len_list = [256, 1024, 4096, 8192]
    results = []

    for d_model, seq_len in tqdm.tqdm(itertools.product(d_model_list, seq_len_list)):
        # Warm-up
        Q = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)
        for _ in range(5):  # Warm-up runs
            _ = cs336_basics.model.scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()

        # Forward pass
        torch.cuda.reset_peak_memory_stats()
        out = cs336_basics.model.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        memory_forward = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

        # Backward pass
        torch.cuda.reset_peak_memory_stats()
        out.sum().backward()
        torch.cuda.synchronize()
        memory_backward = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

        print(f"d_model={d_model}, seq_len={seq_len}, memory_forward={memory_forward:3f}MB, memory_backward={memory_backward:3f}MB")
        results.append({
            'd_model': d_model,
            'seq_len': seq_len,
            'memory_forward': memory_forward,
            'memory_backward': memory_backward,
            'status': 'success'
        })


    return results

if __name__ == '__main__':
    results = benchmark_attention()
    for res in results:
        print(res)