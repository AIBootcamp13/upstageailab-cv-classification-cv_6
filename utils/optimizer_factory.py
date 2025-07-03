from typing import Iterable

from torch.optim import Adam, AdamW, SGD
from torch import optim
from torch.nn import Module, Parameter


OPTIMIZER_REGISTRY: dict[str, optim.Optimizer] = {
    "Adam": Adam,
    "AdamW": AdamW,
    "SGD": SGD,
}

def get_optimizer(name: str, params_to_update: Iterable[Parameter], params: dict) -> optim.Optimizer:
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZER_REGISTRY[name](params_to_update, **params)


if __name__ == '__main__':
    from torch import nn
    
    # 가상의 모델 생성
    dummy_model = nn.Linear(10, 2) 

    # 1. AdamW 사용 예시
    adamw_params = {
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-2
    }
    optimizer1 = get_optimizer("AdamW", dummy_model, optimizer_params=adamw_params)
    print("AdamW Optimizer:", optimizer1)

    # 2. SGD 사용 예시
    sgd_params = {
        'lr': 0.1,
        'momentum': 0.9,
    }
    optimizer2 = get_optimizer("SGD", dummy_model, optimizer_params=sgd_params)
    print("SGD Optimizer:", optimizer2)