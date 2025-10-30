import torch

from .zero_shot_prompt_templates import imagenet_templates
from .clip import load, tokenize


def get_zeroshot_weights(model_variant, classnames, templates, device):
    model, _ = load(model_variant)
    model.to(device).eval()

    if templates == 'imagenet':
        templates = imagenet_templates
    else:
        templates = ['a photo of a {}']

    classnames = [classname.replace('_', ' ') for classname in classnames]

    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        
    return zeroshot_weights