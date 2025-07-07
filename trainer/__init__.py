__all__ = ["training_loop", "TRAINING_REGISTRY", "setting_sampler"]

from trainer.train_loop import training_loop
from trainer.training import training, training_use_amp
from trainer.weighted_sampler import setting_sampler

TRAINING_REGISTRY = {
  "normal": training,
  "on_amp": training_use_amp
}