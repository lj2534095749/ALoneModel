import torch
import torch.nn as nn


class SENet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SENet, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


if __name__ == "__main__":
    model = SENet(512)
    print(model)
    inputs = torch.ones(2, 512, 26, 26)
    outputs = model(inputs)
    print(outputs.size())
