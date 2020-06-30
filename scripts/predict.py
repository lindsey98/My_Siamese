import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from scripts.model_resnetv2 import ResNetV2
from collections import OrderedDict

KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])

def pred(img_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 128
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    ## normalization
    img_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std),
         ])

    img = Image.open(img_path).convert("RGB")

    ## resize the image while keeping the original aspect ratio
    img = ImageOps.expand(img, (
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
        (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))
    img = img.resize((img_size, img_size))

    ## make prediction --> get the feature vector
    with torch.no_grad():
        img = img_transforms(img)
        img = img[None, ...]
        img = img.to(device)
        logo_feat = model.features(img)
    #         logo_feat = l2_norm(logo_feat).squeeze(0).cpu().numpy()
    return logo_feat


def load_model(classes, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model
    model = KNOWN_MODELS["BiT-M-R50x1"](head_size=classes, zero_head=True)
    # Load weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


if __name__ == '__main__':
    ## do not change these configurations
    classes = 180
    modelpath = 'model/rgb_ar.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(classes, modelpath)
    model.to(device)
    model.eval()

    ## change img_path to the logo you want to test
    img_path = 'targetlist_updated_clean/1&1 Ionos/1.png'
    logo_feat = pred(img_path, model)
