import argparse
import os

import numpy as np
import torch.utils.data
import torchvision
from tensorboardX import SummaryWriter
from torch import nn, optim
from torchvision import datasets
from torchvision.transforms import transforms


IMAGE_SHAPE = (28, 28)
IMAGE_SIZE = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
BOTTLENECK_SIZE = 3
LEARNING_RATE = 0.01
MAX_EPOCHS = 100
WRITE_IMAGE_EVERY_N_EPOCHS = 10
IMG_OUT_DIRECTORY = 'mnist-conv-autoencoder_img'



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class UnFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 64, *(np.array(IMAGE_SHAPE) - 4))



class AutoEncoder(nn.Module):
    def __init__(self, bottleneck_size):
        super(AutoEncoder, self).__init__()

        encoder_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size = 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size = 3),
                nn.ReLU(),
                Flatten()
        )
        conv_out_size = encoder_conv(torch.zeros(1, 1, *IMAGE_SHAPE)).size(1)
        self._encoder = nn.Sequential(
                encoder_conv,
                nn.Linear(conv_out_size, 64),
                nn.ReLU(),
                nn.Linear(64, bottleneck_size),
                nn.ReLU()
        )

        self._decoder = nn.Sequential(
                nn.Linear(bottleneck_size, 64),
                nn.ReLU(),
                nn.Linear(64, conv_out_size),
                nn.ReLU(),
                UnFlatten(),
                nn.ConvTranspose2d(64, 32, kernel_size = 3),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size = 3),
                nn.Tanh()
        )


    def forward(self, x):
        latent = self._encoder(x)
        return self._decoder(latent)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action = 'store_true', help = 'Enable CUDA acceleration.')
    parser.add_argument('--overwrite', action = 'store_true', help = 'Overwrite image directory %s directory if it exists.' % IMG_OUT_DIRECTORY)
    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise Exception('CUDA acceleration requested but is not available!')
    device = torch.device('cuda' if args.cuda else 'cpu')

    if os.path.exists(IMG_OUT_DIRECTORY):
        if args.overwrite:
            os.removedirs(IMG_OUT_DIRECTORY)
        else:
            raise Exception('Image directory %s exists!' % IMG_OUT_DIRECTORY)
    os.makedirs(IMG_OUT_DIRECTORY)

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
        train_loss = 0
        for img, _ in train_data:
            img = img.to(device)
            out = ae(img)

            optimizer.zero_grad()
            loss = loss_fn(out, img)
            train_loss += loss
            loss.backward()
            optimizer.step()
        train_loss /= len(train_data)

        test_loss = 0
        for img, _ in test_data:
            img = img.to(device)
            out = ae(img)

            optimizer.zero_grad()
            loss = loss_fn(out, img)
            test_loss += loss
        test_loss /= len(test_data)

        print('Epoch %5d: train_loss=%.5f, test_loss=%.5f' % (epoch, train_loss, test_loss))
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)

        if epoch % WRITE_IMAGE_EVERY_N_EPOCHS == 0:
            # noinspection PyUnboundLocalVariable
            writer.add_image('original', torchvision.utils.make_grid(img.view(out.size(0), 1, 28, 28)), epoch)
            # noinspection PyUnboundLocalVariable
            writer.add_image('reconstruction', torchvision.utils.make_grid(out.view(out.size(0), 1, 28, 28)), epoch)
            torchvision.utils.save_image(img.view(img.size(0), 1, 28, 28), IMG_OUT_DIRECTORY + '/img_%05d.png' % epoch)
            torchvision.utils.save_image(out.view(out.size(0), 1, 28, 28), IMG_OUT_DIRECTORY + '/out_%05d.png' % epoch)
    writer.close()
