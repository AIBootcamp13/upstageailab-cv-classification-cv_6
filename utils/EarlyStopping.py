import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0.0, save_path='checkpoint.pt', mode='min', enabled=True):
        """
        Args:
            patience (int): 개선이 없을 때 기다릴 에폭 수
            verbose (bool): 개선 시 출력 여부
            delta (float): 개선이라고 간주할 최소 변화량
            save_path (str): 모델을 저장할 경로
            mode (str): 'min' 또는 'max'. min은 지표가 감소할 때, max는 증가할 때를 최적으로 판단.
            enabled (bool): 조기 종료 활성화 여부
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        self.enabled = enabled
        self.counter = 0
        self.early_stop = False

        # ✅ mode에 따라 best_score 초기값 설정
        self.mode = mode
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

    def __call__(self, metric_value, model):
        if not self.enabled:
            return  # 조기 종료를 사용하지 않음

        # ✅ mode에 따라 개선 여부 판단
        is_improvement = False
        if self.mode == 'max':
            if metric_value > self.best_score + self.delta:
                is_improvement = True
        else: # mode == 'min'
            if metric_value < self.best_score - self.delta:
                is_improvement = True

        if is_improvement:
            self.best_score = metric_value
            self._save_checkpoint(metric_value, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  ↪️ No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, metric_value, model):
        '''최적의 모델을 저장'''
        if self.verbose:
            # ✅ mode에 따라 개선 메시지 출력
            print(f"  ✅ Validation metric improved ({self.best_score:.4f}). Saving model...")
        torch.save(model.state_dict(), self.save_path)