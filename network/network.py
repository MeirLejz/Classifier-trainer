import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, numChannels: int = 1, numClasses: int = 10):
        super(Classifier, self).__init__()
        self.convo_net = nn.Sequential(
            nn.Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.Flatten(),
            nn.Linear(in_features=50*4*4, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=numClasses),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.convo_net(x)
        return output