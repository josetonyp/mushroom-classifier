from .efficient_net_b1_base import EfficientNetB1Base
from .efficient_net_b5_base import EfficientNetB5Base
from .efficient_net_b7_base import EfficientNetB7Base
from .resnet50_base import ResNet50Base
from .vgg16_base import VGG16Base
from .vgg19_base import VGG19Base


class Factory:
    def build(base, file_size):
        match base:
            case "vgg16":
                return VGG16Base(file_size)
            case "vgg19":
                return VGG19Base(file_size)
            case "resnet50":
                return ResNet50Base(file_size)
            case "efficientNetB1":
                return EfficientNetB1Base(file_size)
            case "efficientNetB5":
                return EfficientNetB5Base(file_size)
            case "efficientNetB7":
                return EfficientNetB7Base(file_size)
            case _:
                raise ValueError("Please select an existing model")
