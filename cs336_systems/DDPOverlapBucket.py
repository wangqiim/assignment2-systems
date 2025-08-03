import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter
from typing import List, Dict, Tuple
from enum import Enum, auto

class GradientState(Enum):
    """Enum representing the state of a parameter's gradient during training."""
    NOT_READY = auto()   # Gradient not yet computed
    READY = auto()       # Gradient computed but not synchronized
    REDUCED = auto()   # Gradient has been synchronized
    
class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        """
        Args:
            module: The model to wrap with DDP
            bucket_size_mb: Maximum bucket size in megabytes
        """
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.buckets: List[List[Tuple]] = []
        self._param_to_bucket_idx: Dict[Parameter, int] = {}
        
        self._create_buckets()
        self._synchronize_initial_weights()
        self._register_hooks()
        
        self.comm_handles = []
        self._grad_states: Dict[Parameter, GradientState] = {}
        
    def _create_buckets(self):
        """Organize parameters into buckets based on specified size."""
        current_bucket = []
        current_size = 0  # in bytes
        
        # Process parameters in reverse order (gradients become ready in reverse)
        for name, param in reversed(list(self.module.named_parameters())):
            if not param.requires_grad:
                continue
            param_size = param.numel() * param.element_size()
            
            # Start new bucket if current one would exceed size limit
            if current_bucket and (current_size + param_size) > (self.bucket_size_mb * 1024 * 1024):
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
                
            current_bucket.append((name, param))
            current_size += param_size
            
            # Map parameter to its bucket index
            self._param_to_bucket_idx[param] = len(self.buckets)
            
        # Add the last bucket if not empty
        if current_bucket:
            self.buckets.append(current_bucket)
        
    def _synchronize_initial_weights(self):
        """Ensure all ranks start with the same parameters."""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)
    
    
    def _grad_reduce_hook(self, name) -> None:
        def fn(param):
            self._grad_states[param] = GradientState.READY
            self._maybe_launch_bucket_comm(name, param)
        return fn
    
    def _register_hooks(self):
        """Register backward hooks to trigger gradient synchronization when buckets are ready."""
        self._grad_states = {param: GradientState.NOT_READY for param in self.module.parameters()}

        for bucket in self.buckets:
            for name, param in bucket:
                if param.requires_grad:
                    # if dist.get_rank() ==0:
                    #     print(f"register hook: {name}!!!!")
                    param.register_post_accumulate_grad_hook(self._grad_reduce_hook(name))
                
    def _maybe_launch_bucket_comm(self, name, trigger_param):
        """Check if any buckets are ready for communication and launch all-reduce."""
        bucket_idx = self._param_to_bucket_idx[trigger_param]
        bucket = self.buckets[bucket_idx]
        # if dist.get_rank() == 0:
        #     print(f"param: {name}, bucket_idx = {bucket_idx}, bucket size = {len(bucket)}")
        
        if all(self._grad_states[p] == GradientState.READY for name, p in bucket):
            # Launch all-reduce asynchronously
            
            grads = []
            i = 0
            for name, p in bucket:
                self._grad_states[p] = GradientState.REDUCED
                p.grad /= dist.get_world_size()
                grads.append(p.grad)
                i += 1
                # if dist.get_rank() == 0:
                #     print("=================before =============")
                #     print(f"====bucket{bucket_idx},i{i}==============")
                #     print("==============================")
                
            flat_grad = torch._utils._flatten_dense_tensors(grads)
            
            handle = dist.all_reduce(flat_grad, async_op=True)
            self.comm_handles.append((handle, flat_grad, grads, bucket_idx))
                
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
        
    def finish_gradient_synchronization(self):
        """Wait for all asynchronous communication to complete."""
        # Wait for all communication and unflatten results
        for handle, flat_grad, grads, bucket_idx in self.comm_handles:
            i = 0
            handle.wait()
            reduced_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)
            for i in range(len(grads)):
                grads[i].copy_(reduced_grads[i])
                i += 1
            # if dist.get_rank() == 0:
            #     print(f"======= after {bucket_idx}========")
            #     print(f"===========i{i}==============")
            #     print("==============================")
            
        # Reset for next iteration
        self.comm_handles = []
        for param in self.module.parameters():
            self._grad_states[param] = GradientState.NOT_READY
