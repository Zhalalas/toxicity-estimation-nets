from torch import nn

class FirstNetwork(nn.Module):
    def __init__(self):
        super(FirstNetwork, self).__init__()
        self.layer1 = nn.Linear(5, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 2)
        self.tanh = nn.Tanh()
        self.softmax = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.tanh.forward(x)
        x = self.layer2.forward(x)
        x = self.tanh.forward(x)
        x = self.layer3.forward(x)
        x = self.tanh.forward(x)
        x = self.output_layer.forward(x)
        x = self.softmax.forward(x)
        return x
