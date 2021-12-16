from torch import nn

# Define CNN arch
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.PReLU(20),
            nn.AvgPool2d((2,2), stride=2),

            nn.Conv2d(20, 40, 5),
            nn.PReLU(40),
            nn.AvgPool2d((2,2), stride=2),

            nn.Conv2d(40, 80, 5),
            nn.PReLU(80),
            nn.AvgPool2d((2,2), stride=2),
            nn.BatchNorm2d(80),
            nn.Dropout2d(0.25),

            nn.Conv2d(80, 80, 2),
            nn.PReLU(80),

            nn.Conv2d(80, 160, 1),
            nn.PReLU(160),

            nn.Conv2d(160, 160, 1),
            nn.PReLU(160)

        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 160),
            nn.PReLU(160),
            nn.Dropout2d(0.5),

            nn.Linear(160, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        logits = self.linear(x)
        return logits
