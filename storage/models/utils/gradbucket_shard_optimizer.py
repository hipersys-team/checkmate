from typing import Any, Callable, Optional, Type

from torch.optim import Optimizer


class GradBucketShardedOptimizer(Optimizer):
  ''' 
  An optimizer that is sharded according to the grad buckets.
  For now this runs locally. 
  '''
  def __init__(
    self,
    local_grad_buckets,
    optimizer_class: Type[Optimizer],
    **defaults: Any,
  ):
    params = [p for bucket in self.local_grad_buckets for p in bucket.parameters()]
    Optimizer.__init__(self, params, defaults)
    self.local_grad_buckets = local_grad_buckets
    self.__optim_defaults = defaults
    self.local_optimizer = optimizer_class(
      params=params,
      **self.__optim_defaults
    )
    
  def zero_grad(self, set_to_none: bool = True) -> None:
    return super().zero_grad(set_to_none)
  
  def step(
      self,
      closure: Optional[Callable[[], float]] = None,
      **kwargs: Any,
  ) -> Optional[float]:
    return self.local_optimizer.step(closure, **kwargs)
  
  