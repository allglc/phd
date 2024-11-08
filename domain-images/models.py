from torch import nn

class CNN_MNIST(nn.Module):
    ''' from https://github.com/pytorch/examples/blob/master/mnist/main.py'''
    def __init__(self, image_channels=1, output_dim=10):
        super().__init__()
        kernel_size = 3
        stride = 1
        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size, stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size, stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)