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



class Encoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(Encoder, self).__init__()

        self._fc = nn.Sequential(
                nn.Linear(IMAGE_SIZE, 128),
                nn.ReLU(True),
                nn.Linear(128, 64),
                nn.ReLU(True),
                nn.Linear(64, 12),
                nn.ReLU(True),
                nn.Linear(12, bottleneck_size),
                nn.ReLU(True)
        )


    def forward(self, x):
        return self._fc(x)



class Decoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(Decoder, self).__init__()

        self._fc = nn.Sequential(
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
        return self._fc(x)



class AutoEncoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(bottleneck_size)
        self.decoder = Decoder(bottleneck_size)


    def forward(self, x):
        return self.decoder(self.encoder(x))



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
    epoch = 0
    while True:
        epoch += 1

        losses = []
        for img, _ in train_data:
            img = img.view(img.size(0), -1)
            img = img.to(device)
            out = ae(img)

            optimizer.zero_grad()
            loss = loss_fn(out, img)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        mean_loss = float(np.mean(losses))

        print('Epoch %5d: mean_loss=%.5f' % (epoch, mean_loss))
        writer.add_scalar('mean_loss', mean_loss, epoch)

        torchvision.utils.save_image(img.view(img.size(0), 1, 28, 28), '../img/img_%05d.png' % epoch)
        torchvision.utils.save_image(out.view(out.size(0), 1, 28, 28), '../img/out_%05d.png' % epoch)
