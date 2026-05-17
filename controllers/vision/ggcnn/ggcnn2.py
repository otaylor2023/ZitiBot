"""GG-CNN2 model.

Vendored verbatim (with minor formatting) from
https://github.com/dougsm/ggcnn/blob/master/models/ggcnn2.py
so the pretrained `*_statedict.pt` weights load with strict key matching.

Original work: Douglas Morrison, Peter Corke, Juergen Leitner,
"Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp
Synthesis Approach", RSS 2018.

License: BSD-3-Clause (upstream).
"""

from __future__ import annotations

import torch.nn as nn


class GGCNN2(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        filter_sizes: list[int] | None = None,
        l3_k_size: int = 5,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [
                16,  # First set of convs
                16,  # Second set of convs
                32,  # Dilated convs
                16,  # Transpose Convs
            ]
        if dilations is None:
            dilations = [2, 4]

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, filter_sizes[0], kernel_size=11, stride=1, padding=5, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[0], filter_sizes[0], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_sizes[1], filter_sizes[1], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                filter_sizes[1], filter_sizes[2],
                kernel_size=l3_k_size, dilation=dilations[0], stride=1,
                padding=(l3_k_size // 2 * dilations[0]), bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                filter_sizes[2], filter_sizes[2],
                kernel_size=l3_k_size, dilation=dilations[1], stride=1,
                padding=(l3_k_size // 2 * dilations[1]), bias=True,
            ),
            nn.ReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[2], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(filter_sizes[3], filter_sizes[3], 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.cos_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.sin_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)
        self.width_output = nn.Conv2d(filter_sizes[3], 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.features(x)
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        return pos_output, cos_output, sin_output, width_output
