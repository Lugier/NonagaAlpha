import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        return F.relu(x)


class NonagaNet(nn.Module):
    """
    A CNN for evaluating Nonaga states, optimized for NVIDIA RTX 3090 / High-end GPUs.
    Input shape: (B, 5, 19, 19)
      Channels:
      0: 1 if cell is inside the current 19-disc board, else 0
      1: 1 if Red piece is on cell, else 0
      2: 1 if Black piece is on cell, else 0
      3: 1 if cell is the forbidden disc, else 0
      4: 1 if Red to move, -1 if Black to move
    
    Outputs:
      - Policy Logits: (B, 4, 19, 19) corresponding to (Slide Start, Slide End, Tile Remove, Tile Place)
      - Value: (B, 1) in [-1, 1] evaluation from the perspective of the current player.
    """
    def __init__(self, num_res_blocks: int = 10, num_channels: int = 128):
        super().__init__()
        self.conv_in = nn.Conv2d(5, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)
        
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_res_blocks)]
        )
        
        self.pol_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.pol_bn = nn.BatchNorm2d(32)
        self.pol_fc = nn.Linear(32 * 19 * 19, 4 * 19 * 19)
        
        self.val_conv = nn.Conv2d(num_channels, 3, kernel_size=1, bias=False)
        self.val_bn = nn.BatchNorm2d(3)
        self.val_fc1 = nn.Linear(3 * 19 * 19, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        
        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = p.flatten(1)
        pi_logits = self.pol_fc(p)
        pi_logits = pi_logits.view(-1, 4, 19, 19)
        
        v = F.relu(self.val_bn(self.val_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.val_fc1(v))
        val = torch.tanh(self.val_fc2(v))
        
        return pi_logits, val


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
