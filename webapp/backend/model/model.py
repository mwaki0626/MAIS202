import sys
sys.path.append(r"/Users/monamiwaki/Fall2020-Workshop3/backend/model")
from .nn import Net

import matplotlib
matplotlib.use('TkAgg')
import torch
import torchvision
import torch.optim as optim

size = (28, 28)


def load_data(batch_size_train=64, batch_size_test=1000):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader


class ImageDetectModel:
    def __init__(self, learning_rate=0.01, momentum=0.5):
        self.net = Net()
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=momentum)

    def train_model(self, train_data, test_data, n_epochs=3, log_interval=100):
        '''
        This method trains the model at a higher level. Calls train and test methods.
        '''

        for epoch in range(1, n_epochs + 1):
            model.train(train_data, epoch, log_interval)
            model.test(test_data)



if __name__ == "__main__":
    model = ImageDetectModel()
    random_seed = 1
    torch.manual_seed(random_seed)

    # Load the data.
    train_loader, test_loader = load_data()

    # Train the model
    model.train_model(train_loader, test_loader, 6)

