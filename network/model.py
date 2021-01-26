from torch import  nn
import torch


class Generator(nn.Module):
    def __init__(self, z_size=16, n_channels=1):
        super(Generator, self).__init__()
        self.n_channels = n_channels
        self.z_size = z_size
        dim = z_size
        preprocess = nn.Sequential(
            nn.Linear(z_size, 4 * 4 * 4 * dim),
            nn.BatchNorm1d(4 * 4 * 4 * dim),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, stride=2),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(True),
            nn.Conv2d(2 * dim, 2 * dim, 3,1,1),
            nn.BatchNorm2d(2*dim),
            nn.ReLU(True),
        ) #out 2*dim,8,8
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3,1,1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        ) #out 2*dim,16,1
        deconv_out = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, n_channels, 3,1,1)
        ) #out nc,32,32
        

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(len(output), -1, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)
        return output.view(-1, self.n_channels, 32, 32)


class Discriminator(nn.Module):
    def __init__(self, dim=16, n_channels=1):
        super(Discriminator, self).__init__()
        self.n_channels = n_channels
        self.dim = dim
        main = nn.Sequential(
            nn.Conv2d(n_channels, dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
        )
        self.main = main
        self.linear = nn.Linear(4*4*4*dim, 1)

    def forward(self, input, return_feat=False):
        output = self.main(input)
        output = output.view(len(output), -1)
        if return_feat:
            output
        output = self.linear(output)
        return output


class Encoder(nn.Module):
     def __init__(self,dim,z_size, n_channels):
        super(Encoder, self).__init__()
        # self.dim = dim
        main = nn.Sequential(
            nn.Conv2d(n_channels, dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2*dim, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2*dim, 4*dim, 3, 1, padding=1),
            nn.LeakyReLU(),
        )
        self.main = main
        self.linear = nn.Linear(4*4*4*dim, z_size)

     def forward(self, input):
        output = self.main(input)
        output = output.view((len(output), -1))
        output = self.linear(output)
        return output

if __name__ == '__main__':
    pass