from typing import Tuple

import torch
import torch.nn as nn


class EnhanceNetNoPool(nn.Module):
    """Zero-DCE enhancement network without pooling layers.

    Predicts pixel-wise illumination adjustment curves from a low-light
    input, producing an enhanced image in a single forward pass.

    Input:  (B, 3, H, W) tensor in [0, 1]
    Output: tuple of (enhance_image_1, enhance_image, r) where:
      - enhance_image_1: intermediate enhanced image
      - enhance_image: final enhanced image
      - r: predicted curve parameter maps (B, 24, H, W)
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        nf = 32  # number of filters per conv layer
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image_1, enhance_image, r



