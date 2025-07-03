__all__ = ['TRANSFORM_REGISTRY', 'load_transforms_from_yaml']
import numpy as np
from PIL import Image
import yaml
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.autoaugment import AutoAugmentPolicy
import augraphy as ag

from autoAugment.autoaugment import ImageNetPolicy, SVHNPolicy, CIFAR10Policy

__all__ = ['load_transforms_from_yaml']

# 1. 3개 라이브러리의 변환 함수/클래스를 등록
TORCHVISION_REGISTRY = {
    "Resize": T.Resize,
    "CenterCrop": T.CenterCrop,
    "RandomResizedCrop": T.RandomResizedCrop,
    "RandomHorizontalFlip": T.RandomHorizontalFlip,
    "ToTensor": T.ToTensor,
    "Normalize": T.Normalize,
    "RandomRotation": T.RandomRotation,
}

ALBUMENTATIONS_REGISTRY = {
    "Resize": A.Resize,
    "HorizontalFlip": A.HorizontalFlip,
    "VerticalFlip": A.VerticalFlip,
    "ShiftScaleRotate": A.ShiftScaleRotate,
    "RandomBrightnessContrast": A.RandomBrightnessContrast,
    "Normalize": A.Normalize,
    "RandomRotate90": A.RandomRotate90,
    "ColorJitter": A.ColorJitter,
    "CenterCrop": A.CenterCrop,
}

AUGRAPHY_REGISTRY = {
    "InkBleed": ag.InkBleed,
    "LowInkRandomLines": ag.LowInkRandomLines,
    "BleedThrough": ag.BleedThrough,
    "DirtyRollers": ag.DirtyRollers,
    "Jpeg": ag.Jpeg,
    "OneOf": ag.OneOf,
    "DirtyDrum": ag.DirtyDrum,
    "LowLightNoise": ag.LowLightNoise,
    "LightingGradient": ag.LightingGradient,
    "BadPhotoCopy": ag.BadPhotoCopy,
    "ShadowCast": ag.ShadowCast,
    "NoiseTexturize": ag.NoiseTexturize,
}

ALL_REGISTRIES = {
    "torchvision": TORCHVISION_REGISTRY,
    "albumentations": ALBUMENTATIONS_REGISTRY,
    "augraphy": AUGRAPHY_REGISTRY,
}


# 2. 데이터 타입을 자동으로 변환해주는 통합 Compose 클래스
class UnifiedCompose:
    """
    Augraphy, Albumentations(NumPy기반)와 Torchvision(PIL기반) 변환을
    함께 사용할 수 있도록 데이터 타입을 자동으로 변환해주는 Compose 클래스.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        # 최초 입력은 NumPy 배열이라고 가정
        is_pil = False
        original_img = img

        for t in self.transforms:
            img_before_transform = img
            backend = getattr(t, '__module__', '').split('.')[0]
            
            # --- 데이터 타입 변환 로직 (이전과 동일) ---
            if backend == 'torchvision' and not isinstance(t, (T.ToTensor, T.Normalize)):
                if not is_pil:
                    img = Image.fromarray(np.uint8(img))
                    is_pil = True
            elif backend in ['augraphy', 'albumentations']:
                if is_pil:
                    img = np.array(img)
                    is_pil = False
            
            if isinstance(t, T.ToTensor):
                 is_pil = False
            
            # --- 변환 적용 로직 (수정된 부분) ---
            if img is not None:
                if backend == 'albumentations':
                    img = t(image=img)['image']
                else: # augraphy 또는 torchvision
                    output = t(img)
                    
                    # Augraphy 등이 튜플을 반환할 경우를 처리
                    if isinstance(output, tuple):
                        # 튜플의 첫 번째 요소를 이미지라고 가정
                        img = output[0]
                    else:
                        # 튜플이 아니면 결과 그대로 사용
                        img = output
            
            # --- 방어 코드 (이전과 동일) ---
            if img is None:
                img = img_before_transform
                is_pil = isinstance(img, Image.Image)

        if img is None:
            img = original_img

        return img
    
    def __iter__(self):
        """이 클래스가 for 루프에서 사용될 수 있도록 내부의 transform 리스트에 대한 반복자(iterator)를 반환합니다."""
        return iter(self.transforms)
    
    def __getitem__(self, index):
        """이 클래스가 my_object[index]와 같은 인덱싱을 지원하도록 합니다."""
        return self.transforms[index]

# 3. YAML 리스트로부터 통합 파이프라인을 만드는 빌더 함수
def build_unified_transforms(transform_list):
    if not transform_list:
        return None
        
    ops = []
    for item in transform_list:
        backend = item["backend"]
        name = item["name"]
        params = item.get("params", {}).copy()
        
        transform_cls = ALL_REGISTRIES[backend][name]
        
        # 'OneOf' 같은 중첩 구조를 재귀적으로 처리
        if "transforms" in params:
            nested_ops_config = params.pop("transforms")
            nested_ops = build_unified_transforms(nested_ops_config) # 재귀 호출
            ops.append(transform_cls(nested_ops, **params))
        else:
            ops.append(transform_cls(**params))
            
    return UnifiedCompose(ops)


# 4. YAML 파일로부터 변환 파이프라인을 로드하는 메인 함수
def load_transforms_from_yaml(yaml_path):
    """
    YAML 파일 경로를 받아 학습 및 검증용 변환 파이프라인을 생성합니다.
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)["transforms"]
        
    train_tf = build_unified_transforms(cfg.get("train"))
    val_tf = build_unified_transforms(cfg.get("val"))

    return train_tf, val_tf