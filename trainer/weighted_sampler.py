import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def setting_sampler(train_path: str) -> WeightedRandomSampler:
    train_df = pd.read_csv(train_path)
    
    # 각 클래스별 데이터 개수 계산
    class_counts = train_df['target'].value_counts().sort_index()
    
    # 클래스별 가중치 계산 (개수의 역수)
    class_weights = 1.0 / class_counts
    
    # 데이터프레임의 'target' (레이블) 컬럼을 기반으로 각 샘플의 가중치를 가져옴
    sample_weights = train_df['target'].map(class_weights).to_numpy()
    
    # PyTorch 텐서로 변환
    sample_weights = torch.from_numpy(sample_weights).double()
    
    return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # 복원 추출 허용
        )