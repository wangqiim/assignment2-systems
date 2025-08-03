import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type, Any, Dict, List, Optional

class ShardedOptimizer(Optimizer):
    """
    A wrapper around a PyTorch optimizer that shards optimizer states across multiple ranks.
    """
    
    def __init__(
        self,
        params,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any
    ):
        # Store the optimizer class and its kwargs
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        
        # Get the world size and rank
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.optimizer = None
        # 这里非常坑，Optimizer.__init__时，必须传递参数值，不能为空，同时__init__内部会调用虚函数add_param_group,
        # 如果在此之前构造optmizer，会导致构造optmizer时候，传递了一遍参数，调用add_param_group时，又传递了一遍参数!!
        # 因此懒构造optimizer，即在第一次add_param_group时同时构造出optimizer
        super().__init__(params, kwargs) # note: call add_param_group internal!!!!!!
    
    def _shard_parameters(self, params) -> List[List[Dict[str, Any]]]:
        """
        Shard parameters across ranks.
        """
        # print("===================================")
        # for param in params:
        #     print("param = ", param)
        # print("===================================")
        # assert isinstance(params, list) and all(isinstance(p, torch.Tensor) for p in params)
        
        # Group parameters by rank
        param_groups = [{"params": []} for _ in range(self.world_size)]
        
        for i, param in enumerate(params):
            rank = i % self.world_size
            param_groups[rank]["params"].append(param)
        
        return param_groups
    
    def step(self, closure=None, **kwargs):
        """
        Perform a single optimization step and synchronize parameters.
        """
        # Run the optimizer step
        loss = self.optimizer.step(closure=closure, **kwargs)
        
        # Synchronize parameters across ranks
        for rank in range(self.world_size):
            for param in self.param_groups[rank]["params"]:
                dist.broadcast(param.data, src=rank)
        
        return loss
    
    def add_param_group(self, param_group: Dict[str, Any]):
        """
        Add a new parameter group to the optimizer.
        """
        if len(self.param_groups) == 0:
            self.param_groups = [[] for _ in range(self.world_size)]
        rank_param_num = len(self.param_groups[self.rank])
        # Update local param_groups
        self.param_groups = self._shard_parameters(
            [p for group in self.param_groups for p in group] + param_group["params"]
        )
        
        # Add the group to the optimizer
        if self.optimizer is None: # 懒构造
            self.optimizer = self.optimizer_cls(self.param_groups[self.rank]["params"], **self.kwargs)
        else:
            self.optimizer.add_param_group({"params": [param for param in self.param_groups[self.rank]["params"][rank_param_num:]]})