import argparse
import os
import shutil

import torch.utils.data
import torchvision
from torch import nn, optim
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms


torch.manual_seed(42)

BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
WRITE_IMAGE_EVERY_N_EPOCHS = 10
NAME = os.path.basename(__file__).replace('.py', '')



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class UnFlatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), 8, 2, 2)



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self._encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size = 3, stride = 3, padding = 1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size = 2, stride = 2),
                nn.Conv2d(16, 8, kernel_size = 3, stride = 2, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2, stride = 1)
        )

        self._decoder = nn.Sequential(
                nn.ConvTranspose2d(8, 16, kernel_size = 3, stride = 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, kernel_size = 5, stride = 3, padding = 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 1, kernel_size = 2, stride = 2, padding = 1),
                nn.Tanh()
        )


    def forward(self, x):
        latent = self._encoder(x)
        return self._decoder(latent)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', action = 'store_true', help = 'Enable CUDA acceleration.')
    parser.add_argument('-f', '--overwrite', action = 'store_true', help = f'Overwrite image directory directory if it exists.')
    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise Exception('CUDA acceleration requested but is not available!')
    device = torch.device('cuda' if args.cuda else 'cpu')
    img_out_directory = f'tmp_{NAME}_img'
    tb_comment = f'-{NAME}'

    if os.path.exists(img_out_directory):
        if args.overwrite:
            shutil.rmtree(img_out_directory)
        else:
            raise Exception('Image directory %s exists!' % img_out_directory)
    os.makedirs(img_out_directory)

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

    ae = AutoEncoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr = LEARNING_RATE, weight_decay = 1e-5)
    writer = SummaryWriter(comment = tb_comment)
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

        with torch.no_grad():
            test_loss = 0
            for img, _ in test_data:
                img = img.to(device)
                out = ae(img)

                test_loss += loss_fn(out, img)
            test_loss /= len(test_data)

        print('Epoch %5d: train_loss=%.5f, test_loss=%.5f' % (epoch, train_loss, test_loss))
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)

        if epoch % WRITE_IMAGE_EVERY_N_EPOCHS == 0:
            # noinspection PyUnboundLocalVariable
            writer.add_image('original', torchvision.utils.make_grid(img.view(out.size(0), 1, 28, 28)), epoch)
            # noinspection PyUnboundLocalVariable
            writer.add_image('reconstruction', torchvision.utils.make_grid(out.view(out.size(0), 1, 28, 28)), epoch)
            torchvision.utils.save_image(img.view(img.size(0), 1, 28, 28), f'{img_out_directory}/img_%05d.png' % epoch)
            torchvision.utils.save_image(out.view(out.size(0), 1, 28, 28), f'{img_out_directory}/out_%05d.png' % epoch)
    writer.close()

    torch.save(ae.state_dict(), f'{NAME}.model')
