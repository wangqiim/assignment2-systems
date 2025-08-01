import timeit
import cs336_basics
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, einsum
from contextlib import nullcontext
import einx
import argparse
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
import math
import torch.cuda.nvtx as nvtx

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    torch.cuda.synchronize()
    with nvtx.range("computing attention scores"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    torch.cuda.synchronize()
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    
    torch.cuda.synchronize()
    with nvtx.range("computing softmax"):
        attention_weights = cs336_basics.model.softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    torch.cuda.synchronize()
    
    with nvtx.range("final matmul"):
        proj = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return proj 

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


class RandomVocabDataset(Dataset):
    def __init__(self, dataset_len, context_length, vocab_size):
      self.dataset_len = dataset_len
      self.context_length = context_length
      self.vocab_size = vocab_size
      self.data = torch.randint(vocab_size, (dataset_len,), dtype=torch.int64)
    
    def __getitem__(self, idx):
      inputs = self.data[idx:idx + self.context_length]
      targets = self.data[idx + 1:idx + 1 + self.context_length]
      return inputs, targets
    
    def __len__(self):
      return self.dataset_len - self.context_length - 1

def benchmark_model(model, dataloader: DataLoader, optimizer, device, num_warmup=5, num_repeats=10, mixed_precision=False, record_memory=False):
  data_iter = iter(dataloader)
  # Warmup
  for _ in range(num_warmup):
    with nvtx.range("WarUp"):
      inputs, targets = next(data_iter)
      inputs = inputs.to(device)
      targets = inputs.to(device)
      logistics = model(inputs)
      loss = cs336_basics.cross_entropy(logistics, targets)
      loss.backward()
  torch.cuda.synchronize()
  
  
  model.eval()
  # Benchmark
  # 1. only forward
  if (record_memory):
    torch.cuda.memory._record_memory_history(max_entries=1000000)
  start_time = timeit.default_timer()
  for _ in range(num_repeats):
    inputs, targets = next(data_iter)
    inputs = inputs.to(device)
    targets = inputs.to(device)
    torch.cuda.synchronize()
    with nvtx.range("Forward"):
      with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
        logistics = model(inputs)
        torch.cuda.synchronize()
  avg_forward_time = (timeit.default_timer() - start_time) / num_repeats
  if (record_memory):
    torch.cuda.memory._dump_snapshot("forward_memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    
  model.train()
  # 2. forward + optimizer
  if (record_memory):
    torch.cuda.memory._record_memory_history(max_entries=1000000)
  start_time = timeit.default_timer()
  for _ in range(num_repeats):
    with nvtx.range("Optimizer"):
      optimizer.zero_grad()
      inputs, targets = next(data_iter)
      inputs = inputs.to(device)
      targets = inputs.to(device)
      with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mixed_precision):
        logistics = model(inputs)
        loss = cs336_basics.cross_entropy(logistics, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
  avg_forward_backward_time = (timeit.default_timer() - start_time) / num_repeats
  if (record_memory):
    torch.cuda.memory._dump_snapshot("forward_backward_memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
  
  return avg_forward_time, avg_forward_backward_time


# 4. 定义 Submitit 作业
class BenchmarkJob:
  def __init__(self, model_config, mixed_precision: bool, record_memory: bool, jit: bool):
    self.model_config = model_config
    self.batch_size=4
    self.context_length = 256
    self.rope_theta = 1000
    self.vocab_size = 10000
    self.mixed_precision = mixed_precision
    self.record_memory = record_memory
    self.jit = jit
    
  def __call__(self):
    device = torch.device("cuda")
    # 初始化模型和数据
    
    net = cs336_basics.BasicsTransformerLM(
      vocab_size=self.vocab_size,
      context_length=self.context_length,
      d_model=self.model_config["d_model"],
      num_layers=self.model_config["num_layers"],
      num_heads=self.model_config["num_heads"],
      d_ff=self.model_config["d_ff"],
      rope_theta=self.rope_theta).to(device)
    if self.jit:
      net = torch.compile(net)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = cs336_basics.AdamW(params, lr=0.01, betas = (0.9, 0.999), eps=1e-8, weight_decay=0.01)
    
    dataset = RandomVocabDataset(dataset_len=self.context_length*100, context_length=self.context_length, vocab_size=self.vocab_size)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    # 运行 benchmark
    avg_forward_time, avg_forward_backward_time = benchmark_model(net, dataloader, optimizer=optimizer, device=device, mixed_precision=self.mixed_precision, record_memory=self.record_memory)
    
    # 返回结果
    return {
      "model_config": self.model_config,
      "benchmark_results": {
        "avg_forward_time": avg_forward_time,
        "avg_forward_backward_time": avg_forward_backward_time
      }
    }

def main():
    parser = argparse.ArgumentParser(description='benchmark 测试')
    parser.add_argument('--mixed_precision', action='store_true', help='启用混合精度模式')
    parser.add_argument('--record_memory', action='store_true', help='记录显存使用情况') # https://docs.pytorch.org/memory_viz
    parser.add_argument('--jit', action='store_true', help='开启jit') # https://docs.pytorch.org/memory_viz
    args = parser.parse_args()
    
    # 要测试的模型配置列表
    model_configs = [
      {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
      # {"d_model": 1024, "d_ff": 3072, "num_layers": 12, "num_heads": 16},
      # {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
      # {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
      # {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
      # {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    ]
    
    # 提交作业
    results = []
    for config in model_configs:
      results.append(BenchmarkJob(config, args.mixed_precision, args.record_memory, args.jit)())
    
    # 打印结果
    for result in results:
      print(f"model config: {result['model_config']}")
      print(f"Average inference time: {result['benchmark_results']['avg_forward_time']:.6f} sec")
      print(f"Average inference and backforward time: {result['benchmark_results']['avg_forward_backward_time']:.6f} sec")
      print("-" * 50)

if __name__ == "__main__":
  main()
