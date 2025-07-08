import torch

from models.Resnet18 import Resnet18Model
from models.Resnet50 import Resnet50Model
from models.Resnet50HeadArc import ResNet50ArcFaceModel
from models.Efficientnet_b3 import EfficientNetB3Model
from models.Efficientnet_b4 import EfficientNetB4Model
from models.Efficientnet_b5 import EfficientNetB5Model
from models.Efficientnet_b3_head_arc import EfficientNetB3ArcFace
from models.Efficientnet_v2_b3 import EfficientNetV2B3Model
from models.Efficientnet_v2_b3_head_arc import EfficientNetV2B3ArcFaceModel
from models.Efficientnet_v2_m_head_arc import EfficientNetV2MArcFaceModel
from models.ResnetSt50_head_arc import ResNeSt50ModelArcFaceModel
from models.ConvNeXt_haed_arc import ConvNeXtArcFaceModel
from models.SwinTransformer_haed_arc import SwinTransformerArcFaceModel
from models.ConvNeXtModel import ConvNeXtModel
from models.ResnetSt101e_head_arc import ResNeSt101eModelArcFaceModel
from models.ResnetSt200e_head_arc import ResNeSt200eModelArcFaceModel

__all__ = ['Resnet18Model', 'Resnet50HeadArc']

MODEL_REGISTRY: 'dict[str, torch.nn.Module]' = {
    "Resnet18Model": Resnet18Model,
    "Resnet50Model": Resnet50Model,
    "ResNet50ArcFaceModel": ResNet50ArcFaceModel,
    "EfficientNetB3Model": EfficientNetB3Model,
    "EfficientNetB4Model": EfficientNetB4Model,
    "EfficientNetB5Model": EfficientNetB5Model,
    "EfficientNetB3ArcFace": EfficientNetB3ArcFace,
    "EfficientNetV2B3Model": EfficientNetV2B3Model,
    "EfficientNetV2B3ArcFaceModel": EfficientNetV2B3ArcFaceModel,
    "EfficientNetV2MArcFaceModel": EfficientNetV2MArcFaceModel,
    "ResNeSt50ModelArcFaceModel": ResNeSt50ModelArcFaceModel,
    "ConvNeXtArcFaceModel": ConvNeXtArcFaceModel,
    "SwinTransformerArcFaceModel": SwinTransformerArcFaceModel,
    "ConvNeXtModel": ConvNeXtModel,
    "ResNeSt101eModelArcFaceModel": ResNeSt101eModelArcFaceModel,
    "ResNeSt200eModelArcFaceModel": ResNeSt200eModelArcFaceModel,
}

def get_model(name: str) -> torch.nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name]