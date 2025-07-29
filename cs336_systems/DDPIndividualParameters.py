import torch
import torch.distributed as dist

class DDPIndividualParameters(torch.nn.Module):
  def __init__(self, module: torch.nn.Module):
    # Given an instantiated PyTorch nn.Module to be parallelized, construct a DDP container that will handle gradient synchronization across ranks. 
    super().__init__()  # 必须调用父类初始化
    self.handles = []
    self.module = module
        
    # 同步初始参数到rank 0的值
    self._sync_initial_parameters()

    # 为每个参数注册梯度hook
    for (name, p) in module.named_parameters():
      if p.requires_grad:
        # print("name: ", name)
        p.register_post_accumulate_grad_hook(self._grad_reduce_hook)


  def _sync_initial_parameters(self):
    """同步所有参数到rank 0的值"""
    for param in self.module.parameters():
      # 确保参数在设备上（GPU训练时）
      
      # Rank 0广播参数给其他rank
      dist.broadcast(param.data, src=0)
        
    # 确保所有rank完成同步
    dist.barrier()

  def _grad_reduce_hook(self, p) -> None:
    p.grad /= dist.get_world_size()
    handle = dist.all_reduce(p.grad, async_op=True)
    self.handles.append(handle)
  
  def forward(self, *inputs, **kwargs): 
    # Calls the wrapped module’s forward() method with the provided positional and keyword arguments.
    return self.module(*inputs, **kwargs)
    
  def finish_gradient_synchronization(self):
    for handle in self.handles:
      handle.wait()
    self.handles.clear()
    

    