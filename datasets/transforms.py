__all__ = ['TRANSFORM_REGISTRY', 'load_transforms_from_yaml']
import importlib
import enum

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
    "Rotate": A.Rotate,
    "ColorJitter": A.ColorJitter,
    "CenterCrop": A.CenterCrop,
    "ImageCompression": A.ImageCompression, # 이미지 품질을 낮춰서 압축 손실을 시뮬레이션함
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
    "Folding": ag.Folding,
    "Geometric": ag.Geometric,
    "SubtleNoise": ag.SubtleNoise,
    
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
        # 최초 입력이 PIL 이면 NumPy로 변환
        if isinstance(img, Image.Image):
            img = np.array(img)
        
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
    

def _resolve_params(params: dict) -> dict:
    """
    파라미터 딕셔너리를 순회하며 '_resolve_' 키가 있는 항목을
    실제 파이썬 객체로 변환하는 범용 헬퍼 함수.
    """
    resolved_params = {}
    for key, value in params.items():
        if isinstance(value, dict) and '_resolve_' in value:
            resolver_info = value['_resolve_']
            resolver_type = resolver_info['type']  # 'enum' 또는 'tuple'
            val_to_resolve = resolver_info['value']

            # ⬇️ type 값에 따라 분기하여 처리
            if resolver_type == "enum":
                class_path_str = resolver_info['class_path']
                try:
                    module_path, class_name = class_path_str.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    resolver_class = getattr(module, class_name)
                except (ImportError, AttributeError) as e:
                    raise ValueError(f"Could not resolve class path: {class_path_str}") from e

                if issubclass(resolver_class, enum.Enum):
                    resolved_params[key] = resolver_class[val_to_resolve]
                else:
                    raise TypeError(f"Class for enum resolver is not an Enum: {resolver_class}")

            elif resolver_type == "tuple":
                # value(리스트)를 튜플 객체로 변환
                if not isinstance(val_to_resolve, list):
                    raise TypeError("Value for tuple resolver must be a list in YAML.")
                resolved_params[key] = tuple(val_to_resolve)

            else:
                raise TypeError(f"Unsupported resolver type: {resolver_type}")
        else:
            # 일반 파라미터는 그대로 복사
            resolved_params[key] = value
            
    return resolved_params


# --- 3. YAML 리스트로부터 통합 파이프라인을 만드는 빌더 함수 (수정된 버전) ---
def build_unified_transforms(transform_list: list):
    """
    YAML 설정으로부터 변환 파이프라인을 만드는 범용 빌더 함수.
    OneOf를 포함한 모든 변환의 파라미터를 동적으로 변환합니다.
    """
    if not transform_list:
        return None
        
    ops = []
    for item in transform_list:
        backend = item["backend"]
        name = item["name"]
        params = item.get("params", {}).copy()

        # 1. 파라미터 변환을 if/else 분기 앞으로 이동하여 모든 경우에 적용
        resolved_params = _resolve_params(params)

        if name == 'OneOf':
            # 2. 이제 resolved_params에서 "transforms" 리스트를 가져와야 함
            nested_ops_config = resolved_params.pop("transforms")
            nested_ops_list = build_unified_transforms(nested_ops_config).transforms
            
            if backend == "albumentations":
                one_of_cls = A.OneOf
            elif backend == "augraphy":
                one_of_cls = ag.OneOf
            else:
                raise ValueError(f"OneOf is not supported for backend: {backend}")
            
            # 3. 변환이 완료된 resolved_params를 OneOf 생성에 사용
            ops.append(one_of_cls(nested_ops_list, **resolved_params))
        
        else: # 일반 변환
            if backend and name:
                transform_cls = ALL_REGISTRIES[backend][name]
                # 4. 여기서도 동일하게 resolved_params를 사용
                ops.append(transform_cls(**resolved_params))
            
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