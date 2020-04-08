import argparse

import numpy as np
import torch.utils.data
import torchvision
from tensorboardX import SummaryWriter
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import transforms


IMAGE_SIZE = 28 * 28
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
BOTTLENECK_SIZE = 3
LEARNING_RATE = 0.01
MAX_EPOCHS = 100



class AutoEncoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(AutoEncoder, self).__init__()

        self._encoder = nn.Sequential(
                nn.Linear(IMAGE_SIZE, 128),
                nn.ReLU(True),
                nn.Linear(128, 64),
                nn.ReLU(True),
                nn.Linear(64, 12),
                nn.ReLU(True),
                nn.Linear(12, bottleneck_size),
                nn.ReLU(True)
        )
        self._decoder = nn.Sequential(
                nn.Linear(bottleneck_size, 12),
                nn.ReLU(True),
                nn.Linear(12, 64),
                nn.ReLU(True),
                nn.Linear(64, 128),
                nn.ReLU(True),
                nn.Linear(128, IMAGE_SIZE),
                nn.Tanh()
        )


    def forward(self, x):
        return self._decoder(self._encoder(x))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action = 'store_true', help = 'Enable CUDA acceleration.')
    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise Exception('CUDA acceleration requested but is not available!')
    device = torch.device('cuda' if args.cuda else 'cpu')

    train_data = torch.utils.data.DataLoader(datasets.MNIST('../data',
                                                            train = True,
                                                            download = True,
                                                            transform = transforms.Compose([
                                                                    transforms.ToTensor()
                                                            ])),
                                             batch_size = BATCH_SIZE,
                                             shuffle = True)
    test_data = torch.utils.data.DataLoader(datasets.MNIST('../data',
                                                           train = False,
                                                           download = False,
                                                           transform = transforms.Compose([
                                                                   transforms.ToTensor()
                                                           ])),
                                            batch_size = TEST_BATCH_SIZE,
                                            shuffle = True)

    ae = AutoEncoder(BOTTLENECK_SIZE).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr = LEARNING_RATE)
    writer = SummaryWriter(comment = '-auto_encoder_mnist')
    for epoch in range(1, MAX_EPOCHS + 1):
        total_loss = 0
        for img, _ in train_data:
            img = img.view(img.size(0), -1)
            img = img.to(device)
            out = ae(img)

            optimizer.zero_grad()
            train_loss = loss_fn(out, img)
            total_loss += train_loss
            train_loss.backward()
            optimizer.step()
        total_loss /= len(train_data)

        print('Epoch %5d: total_loss=%.5f' % (epoch, total_loss))
        writer.add_scalar('total_loss', total_loss, epoch)

        torchvision.utils.save_image(img.view(img.size(0), 1, 28, 28), '../img/img_%05d.png' % epoch)
        torchvision.utils.save_image(out.view(out.size(0), 1, 28, 28), '../img/out_%05d.png' % epoch)
    writer.close()
