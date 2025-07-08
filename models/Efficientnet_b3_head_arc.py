import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.ArcMarginProduct import ArcMarginProduct


class EfficientNetB3ArcFace(nn.Module):
    def __init__(self, num_classes, embedding_size=512, s=30.0, m=0.55):
        """
        EfficientNet-B3를 백본으로 사용하는 ArcFace 모델
        :param num_classes: 최종 분류할 클래스의 개수
        :param embedding_size: 특징을 임베딩할 차원 크기
        """
        super(EfficientNetB3ArcFace, self).__init__()
        
        # 1. Backbone: EfficientNet-B3
        # pretrained=True로 ImageNet 가중치를 가져오고, num_classes=0으로 분류기는 제거합니다.
        self.backbone = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
        
        # EfficientNet-B3의 출력 특징 차원은 1536입니다.
        backbone_output_features = 1536 
        
        # 2. Neck: Backbone의 출력을 ArcFace 헤드로 전달하기 전 차원을 축소하고 정규화합니다.
        self.neck = nn.Sequential(
            nn.Linear(backbone_output_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU()
        )
        
        # 3. Head: ArcFace 헤드
        self.head = ArcMarginProduct(embedding_size, num_classes, s=s, m=m)

    def forward(self, x, labels=None):
        """
        모델의 순전파
        - 학습 시에는 이미지(x)와 정답(labels)을 모두 입력받습니다.
        - 추론 시에는 이미지(x)만 입력받을 수 있습니다.
        """
        # Backbone을 통과시켜 특징 추출
        features = self.backbone(x)
        
        # Neck을 통과시켜 임베딩 벡터 생성
        features = self.neck(features)
        
        # Head를 통과시켜 최종 로짓(logits) 계산
        if labels is not None:
            # 학습: ArcFace 마진 적용
            logits = self.head(features, labels)
        else:
            # 추론: 단순 코사인 유사도 계산 (마진 없음)
            logits = F.linear(F.normalize(features), F.normalize(self.head.weight))
            logits *= self.head.s
            
        return logits


if __name__ == '__main__':
    NUM_CLASSES = 17
    
    # 모델 생성
    model = EfficientNetB3ArcFace(num_classes=NUM_CLASSES)
    model.eval() # 추론 모드로 설정
    
    print("모델이 성공적으로 생성되었습니다.")
    # print(model)

    # 더미 데이터로 테스트
    dummy_images = torch.randn(4, 3, 224, 224) # (배치 크기, 채널, 높이, 너비)
    dummy_labels = torch.randint(0, NUM_CLASSES, (4,)) # (배치 크기,)

    # 1. 학습 시 Forward Pass
    logits_train = model(dummy_images, dummy_labels)
    print(f"학습 시 출력 로짓의 크기: {logits_train.shape}") # 예상: [4, 17]

    # 2. 추론 시 Forward Pass
    logits_eval = model(dummy_images, labels=None)
    print(f"추론 시 출력 로짓의 크기: {logits_eval.shape}") # 예상: [4, 17]