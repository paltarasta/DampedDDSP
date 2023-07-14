import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

#### Large ResNet ####

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time, freq, stride):
        super(ResidualBlock, self).__init__()
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.other_first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, stride), padding=(0,1))
        self.layer_norm1 = nn.LayerNorm([in_channels, time, freq])
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1) #kernel = 1 means no spatial reduction

        self.layer_norm2 = nn.LayerNorm([out_channels//4, time, freq])
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(1,3), stride=(1,stride))

        self.layer_norm3 = nn.LayerNorm([out_channels//4, time, freq-2])
        self.layer_norm4 = nn.LayerNorm([out_channels//4, time, ((freq-3)//stride)+1])
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=(0,1))

        self.residual = nn.Identity()
        self.stride = stride

    def forward(self, x):
      if self.stride == 2:
        res = self.other_first_conv(x)
      else:
        res = self.first_conv(x)

      residual = self.residual(res)

      out = self.layer_norm1(x)
      out = self.relu1(out)
      out = self.conv1(out)

      out = self.layer_norm2(out)
      out = self.relu2(out)
      out = self.conv2(out)
      if self.stride == 2:
        out = self.layer_norm4(out)
      else:
        out = self.layer_norm3(out)
      out = self.relu3(out)
      out = self.conv3(out)

      out += residual

      return out


class ResNet(nn.Module):
    """Residual network."""

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=7, stride=(1,2), padding=3)
        self.maxpool = nn.MaxPool2d((1,3), stride=(1,2))

        self.residual_block1 = ResidualBlock(64, 128, 126, 57, 1)
        self.residual_block2 = ResidualBlock(128, 128, 126, 57, 1)
        self.residual_block3 = ResidualBlock(128, 256, 126, 57, 2)
        self.residual_block4 = ResidualBlock(256, 256, 126, 30, 1)
        self.residual_block5 = ResidualBlock(256, 256, 126, 30, 1)
        self.residual_block6 = ResidualBlock(256, 512, 126, 30, 2)
        self.residual_block7 = ResidualBlock(512, 512, 126, 16, 1)
        self.residual_block8 = ResidualBlock(512, 512, 126, 16, 1)
        self.residual_block9 = ResidualBlock(512, 512, 126, 16, 1)
        self.residual_block10 = ResidualBlock(512, 1024, 126, 16, 2)
        self.residual_block11 = ResidualBlock(1024, 1024, 126, 9, 1)
        self.residual_block12 = ResidualBlock(1024, 1024, 126, 9, 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.maxpool(out)

        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.residual_block7(out)
        out = self.residual_block8(out)
        out = self.residual_block9(out)
        out = self.residual_block10(out)
        out = self.residual_block11(out)
        out = self.residual_block12(out)

        return out
    

#### Small ResNet ####

class SimpleResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time, freq, stride):
        super(SimpleResidualBlock, self).__init__()
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.other_first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, stride), padding=(0,1))
        self.layer_norm1 = nn.LayerNorm([in_channels, time, freq])
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1) #kernel = 1 means no spatial reduction

        self.layer_norm2 = nn.LayerNorm([out_channels//4, time, freq])
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(1,3), stride=(1,stride))

        self.layer_norm3 = nn.LayerNorm([out_channels//4, time, freq-2])
        self.layer_norm4 = nn.LayerNorm([out_channels//4, time, ((freq-3)//stride)+1])
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=(0,1))

        self.residual = nn.Identity()
        self.stride = stride

    def forward(self, x):
      if self.stride == 2:
        res = self.other_first_conv(x)
      else:
        res = self.first_conv(x)

      residual = self.residual(res)

      out = self.layer_norm1(x)
      out = self.relu1(out)
      out = self.conv1(out)

      out = self.layer_norm2(out)
      out = self.relu2(out)
      out = self.conv2(out)
      if self.stride == 2:
        out = self.layer_norm4(out)
      else:
        out = self.layer_norm3(out)
      out = self.relu3(out)
      out = self.conv3(out)

      out += residual

      return out


class SimpleResNet(nn.Module):
    """Residual network."""

    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=7, stride=(1,2), padding=3)
        self.maxpool = nn.MaxPool2d((1,3), stride=(1,2))

        self.residual_block1 = SimpleResidualBlock(64, 128, 126, 57, 1)
        self.conv2 = nn.Conv2d(128, 512, kernel_size=(1,41), stride=(1,2))
        self.residual_block12 = SimpleResidualBlock(512, 1024, 126, 9, 1)


    def forward(self, x):
        out = self.conv(x)
        out = self.maxpool(out)

        out = self.residual_block1(out)
        out = self.conv2(out)
        out = self.residual_block12(out)

        return out
    

class MapToFrequency(nn.Module):

  def __init__(self):

    super(MapToFrequency, self).__init__()

    #self.resnet = ResNet()
    self.resnet = SimpleResNet()
    self.frequency_dense = nn.Linear(9*1024, 100*64)
    self.amplitude_dense = nn.Linear(9*1024, 100)
    self.damping_dense = nn.Linear(9*1024, 100)
    self.register_buffer("scale", torch.logspace(np.log2(20), np.log2(8000), 64, base=2.0))
    self.softmax = nn.Softmax(dim=2)

  def forward(self, x):
    out = self.resnet(x)
    out = rearrange(out, 'z a b c -> z b c a')
    out = torch.reshape(out, (out.shape[0], 126, 9*1024))

    frequency = self.frequency_dense(out)
    print('dense', frequency.shape)
    frequency = rearrange(frequency, 'b t (k f) -> b t k f', f=64)
    frequency = self.softmax(frequency)
    frequency = torch.sum(frequency * self.scale, dim=-1)

    amplitude = self.amplitude_dense(out)
    amplitude = self.softmax(amplitude)
    amplitude = amplitude.squeeze(-1)

    damping = self.damping_dense(out)
    damping = self.softmax(damping)
    damping = damping.squeeze(-1)

    return frequency, amplitude, damping
