import math
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
from torch.nn import Dropout

from .model import VisionTransformer


class PromptedVisionTransformer(VisionTransformer):
    def __init__(self, prompt_config, input_resolution, patch_size, width, layers, heads, output_dim):
        super(PromptedVisionTransformer, self).__init__(input_resolution, patch_size, width, layers, heads, output_dim)

        self.prompt_config = prompt_config

        self.num_tokens = self.prompt_config.NUM_TOKENS
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, width)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = width
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size), 1) + prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")
        
    def train(self, mode=True):
      # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.conv1.eval()
            self.ln_pre.eval()
            self.transformer.eval()
            self.ln_post.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)


    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # incorporate prompt
        B = x.shape[0]
        x = torch.cat((
            x[:,:1,:],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:,1:,:]
        ), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x