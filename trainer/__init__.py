__all__ = ["training_loop", "TRAINING_REGISTRY"]

from trainer.train_loop import training_loop
from trainer.training import training, training_use_amp

TRAINING_REGISTRY = {
  "normal": training,
  "on_amp": training_use_amp
}