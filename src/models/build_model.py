import os
import torch

import src.models.clip as local_clip


def build_model(args):

    # read model state dict
    variant = args.MODEL.VARIANT
    model_path =  local_clip._download(local_clip._MODELS[variant], os.path.expanduser("~/.cache/clip"))
    with open(model_path, 'rb') as opened_file:
        state_dict = torch.jit.load(opened_file, map_location="cpu").eval().state_dict()

    # read model configs
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]

    # initialize models
    model = local_clip.PromptedVisionTransformer(
        args.MODEL.VPT.PROMPT, 
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_width // 64,
        output_dim=embed_dim
    )

    # prune state dict for the prompted vision transformer
    visual_state_dict = {k.replace('visual.', ''): v for k, v in state_dict.items() if 'visual' in k}
    msg = model.load_state_dict(visual_state_dict, strict=False)
    args.logger.debug(msg)
    assert set(msg.missing_keys) == set(['prompt_embeddings']), 'There should be only one missing key: prompt_embeddings'
    assert set(msg.unexpected_keys) == set(), 'There should be no unexpected keys'

    # freeze layers
    for k, p in model.named_parameters():
        if "prompt" not in k:
            p.requires_grad = False
        args.logger.debug('{:<60}: requires_grad={}'.format(k, p.requires_grad))
    
    return model