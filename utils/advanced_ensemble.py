import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


class WeightedEnsemble:
    """
    가중치 기반 앙상블 클래스
    """
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        self.weights = torch.tensor(self.weights, dtype=torch.float32)
        self.weights = self.weights / self.weights.sum()  # 정규화
        
    def predict(self, x, device):
        """
        가중치 기반 앙상블 예측
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                probs = F.softmax(output, dim=1)
                predictions.append(probs)
        
        # 가중치 적용
        weighted_predictions = []
        for i, pred in enumerate(predictions):
            weighted_predictions.append(pred * self.weights[i])
        
        # 가중 평균
        ensemble_prediction = torch.stack(weighted_predictions).sum(dim=0)
        
        return ensemble_prediction
    
    def optimize_weights(self, val_loader, device, num_classes=17):
        """
        검증 데이터를 사용하여 최적 가중치 찾기
        """
        all_predictions = []
        all_labels = []
        
        # 각 모델의 예측 수집
        for model in self.models:
            model.eval()
            model_predictions = []
            
            with torch.no_grad():
                for batch_idx, (images, labels, _) in enumerate(val_loader):
                    images = images.to(device)
                    output = model(images)
                    probs = F.softmax(output, dim=1)
                    model_predictions.append(probs.cpu())
                    
                    if len(all_labels) == 0:  # 첫 번째 모델에서만 라벨 수집
                        all_labels.append(labels)
            
            all_predictions.append(torch.cat(model_predictions))
        
        all_labels = torch.cat(all_labels).numpy()
        
        # 그리드 서치를 통한 가중치 최적화
        best_weights = None
        best_f1 = 0
        
        print("가중치 최적화 중...")
        
        for w1 in np.arange(0.1, 1.0, 0.1):
            for w2 in np.arange(0.1, 1.0, 0.1):
                if len(self.models) == 2:
                    weights = [w1, w2]
                elif len(self.models) == 3:
                    for w3 in np.arange(0.1, 1.0, 0.1):
                        weights = [w1, w2, w3]
                        f1 = self._evaluate_weights(weights, all_predictions, all_labels)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_weights = weights.copy()
                else:
                    # 2개 모델일 때
                    weights = [w1, w2]
                    f1 = self._evaluate_weights(weights, all_predictions, all_labels)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights = weights.copy()
        
        if best_weights is not None:
            self.weights = torch.tensor(best_weights, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()
            print(f"최적 가중치: {self.weights.tolist()}")
            print(f"최적 F1 macro: {best_f1:.4f}")
        
        return best_weights, best_f1
    
    def _evaluate_weights(self, weights, predictions, labels):
        """
        주어진 가중치로 F1 스코어 계산
        """
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()
        
        # 가중 평균 계산
        weighted_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += pred * weights[i]
        
        # 예측 클래스 결정
        predicted_classes = torch.argmax(weighted_pred, dim=1).numpy()
        
        # F1 macro 계산
        f1 = f1_score(labels, predicted_classes, average='macro')
        
        return f1


class StackingEnsemble:
    """
    스태킹 앙상블 클래스
    """
    
    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else LogisticRegression(max_iter=1000)
        self.is_fitted = False
        
    def fit(self, train_loader, val_loader, device, num_classes=17):
        """
        스태킹 앙상블 학습
        """
        print("스태킹 앙상블 학습 시작...")
        
        # 1단계: 기본 모델들의 예측 수집
        train_meta_features, train_labels = self._get_meta_features(train_loader, device)
        val_meta_features, val_labels = self._get_meta_features(val_loader, device)
        
        # 2단계: 메타 모델 학습
        self.meta_model.fit(train_meta_features, train_labels)
        
        # 검증 성능 평가
        val_predictions = self.meta_model.predict(val_meta_features)
        val_f1 = f1_score(val_labels, val_predictions, average='macro')
        val_accuracy = accuracy_score(val_labels, val_predictions)
        
        print(f"스태킹 앙상블 검증 성능:")
        print(f"  F1 Macro: {val_f1:.4f}")
        print(f"  Accuracy: {val_accuracy:.4f}")
        
        self.is_fitted = True
        return val_f1, val_accuracy
    
    def predict(self, x, device):
        """
        스태킹 앙상블 예측
        """
        if not self.is_fitted:
            raise ValueError("스태킹 앙상블이 학습되지 않았습니다.")
        
        # 기본 모델들의 예측 수집
        meta_features = []
        
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                probs = F.softmax(output, dim=1)
                meta_features.append(probs.cpu().numpy())
        
        # 메타 특징 생성
        meta_features = np.concatenate(meta_features, axis=1)
        
        # 메타 모델 예측
        predictions = self.meta_model.predict_proba(meta_features)
        
        return torch.tensor(predictions, dtype=torch.float32)
    
    def _get_meta_features(self, dataloader, device):
        """
        메타 특징 생성
        """
        all_meta_features = []
        all_labels = []
        
        for batch_idx, (images, labels, _) in enumerate(dataloader):
            images = images.to(device)
            batch_meta_features = []
            
            # 각 기본 모델의 예측 수집
            for model in self.base_models:
                model.eval()
                with torch.no_grad():
                    output = model(images)
                    probs = F.softmax(output, dim=1)
                    batch_meta_features.append(probs.cpu().numpy())
            
            # 메타 특징 생성 (모든 모델의 예측을 연결)
            batch_meta_features = np.concatenate(batch_meta_features, axis=1)
            all_meta_features.append(batch_meta_features)
            all_labels.append(labels.numpy())
        
        all_meta_features = np.concatenate(all_meta_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        return all_meta_features, all_labels
    
    def save(self, path):
        """
        메타 모델 저장
        """
        if not self.is_fitted:
            raise ValueError("스태킹 앙상블이 학습되지 않았습니다.")
        
        joblib.dump(self.meta_model, path)
        print(f"스태킹 앙상블 모델이 {path}에 저장되었습니다.")
    
    def load(self, path):
        """
        메타 모델 로드
        """
        if os.path.exists(path):
            self.meta_model = joblib.load(path)
            self.is_fitted = True
            print(f"스태킹 앙상블 모델이 {path}에서 로드되었습니다.")
        else:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")


class VotingEnsemble:
    """
    투표 기반 앙상블 클래스
    """
    
    def __init__(self, models, voting='soft'):
        self.models = models
        self.voting = voting  # 'soft' or 'hard'
        
    def predict(self, x, device):
        """
        투표 기반 앙상블 예측
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                
                if self.voting == 'soft':
                    probs = F.softmax(output, dim=1)
                    predictions.append(probs)
                else:
                    # Hard voting
                    _, predicted = torch.max(output, 1)
                    predictions.append(predicted)
        
        if self.voting == 'soft':
            # 소프트 투표: 확률 평균
            ensemble_prediction = torch.stack(predictions).mean(dim=0)
        else:
            # 하드 투표: 다수결
            predictions = torch.stack(predictions)
            ensemble_prediction = torch.mode(predictions, dim=0)[0]
        
        return ensemble_prediction


class AdaptiveEnsemble:
    """
    적응형 앙상블 - 입력에 따라 모델 가중치 조절
    """
    
    def __init__(self, models, confidence_threshold=0.9):
        self.models = models
        self.confidence_threshold = confidence_threshold
        self.model_confidences = {}
        
    def predict(self, x, device):
        """
        적응형 앙상블 예측
        """
        predictions = []
        confidences = []
        
        # 각 모델의 예측과 신뢰도 수집
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                output = model(x)
                probs = F.softmax(output, dim=1)
                
                # 신뢰도 계산 (최대 확률)
                max_probs = torch.max(probs, dim=1)[0]
                avg_confidence = torch.mean(max_probs).item()
                
                predictions.append(probs)
                confidences.append(avg_confidence)
        
        # 신뢰도 기반 가중치 계산
        weights = torch.tensor(confidences, dtype=torch.float32)
        weights = F.softmax(weights / 0.1, dim=0)  # 온도 스케일링
        
        # 가중 평균
        weighted_predictions = []
        for i, pred in enumerate(predictions):
            weighted_predictions.append(pred * weights[i])
        
        ensemble_prediction = torch.stack(weighted_predictions).sum(dim=0)
        
        return ensemble_prediction


def create_ensemble_predictions(models, test_loader, device, ensemble_type='weighted', 
                              weights=None, use_tta=False, num_tta=5):
    """
    다양한 앙상블 방법으로 예측 생성
    
    Args:
        models: 모델 리스트
        test_loader: 테스트 데이터 로더
        device: 디바이스
        ensemble_type: 앙상블 타입 ('weighted', 'voting', 'adaptive')
        weights: 가중치 (weighted 앙상블 시 사용)
        use_tta: TTA 사용 여부
        num_tta: TTA 개수
    
    Returns:
        앙상블 예측 결과
    """
    if ensemble_type == 'weighted':
        ensemble = WeightedEnsemble(models, weights)
    elif ensemble_type == 'voting':
        ensemble = VotingEnsemble(models, voting='soft')
    elif ensemble_type == 'adaptive':
        ensemble = AdaptiveEnsemble(models)
    else:
        raise ValueError(f"지원하지 않는 앙상블 타입: {ensemble_type}")
    
    all_predictions = []
    all_labels = []
    
    if use_tta:
        from .enhanced_tta import EnhancedTTA
        tta = EnhancedTTA()
    
    for batch_idx, (images, labels, _) in enumerate(test_loader):
        images = images.to(device)
        
        if use_tta:
            # TTA 적용 앙상블
            batch_predictions = []
            for model in models:
                tta_pred = tta.predict_batch_with_tta(model, images, device, num_tta)
                batch_predictions.append(tta_pred)
            
            # 모델 평균
            ensemble_pred = torch.stack(batch_predictions).mean(dim=0)
        else:
            # 일반 앙상블
            ensemble_pred = ensemble.predict(images, device)
        
        all_predictions.append(ensemble_pred.cpu())
        all_labels.append(labels)
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    return all_predictions, all_labels