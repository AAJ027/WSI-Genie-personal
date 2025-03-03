from .timm_wrapper import TimmCNNEncoder
from ..utils.constants import IMAGENET_MEAN, IMAGENET_STD
# from ..utils.transform_utils import get_eval_transforms
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
        
def get_encoder(model_name: str, target_img_size=224):
    
    transforms_list = []
    
    mean = IMAGENET_MEAN
    std = IMAGENET_STD

    transforms_list.append(transforms.Resize(target_img_size))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean, std))
    img_transforms = transforms.Compose(transforms_list)

    
    # print(f"Model requested is {model_name}")
    if model_name.startswith("hibou"):
        if model_name == 'hibou_b':
            model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)
        elif model_name == 'hibou_l':
            model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
        else:
             raise NotImplementedError(f"model {model_name} not found")
    else:
        model = TimmCNNEncoder(model_name=model_name)
    
    return model, img_transforms