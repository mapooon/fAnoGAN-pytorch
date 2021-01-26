from torch import nn
import torch
import torch.autograd as autograd

def calc_gradient_penalty(netD, real_data, fake_data):
    N,C,H,W = real_data.shape
    device = real_data.device
    alpha = torch.rand(N, 1)
    alpha = alpha.expand(N, int(
        real_data.nelement()/N)).contiguous()
    alpha = alpha.view(N, C, H, W)
    alpha = alpha.to(device)

    fake_data = fake_data.view(N, C, H, W)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(
                                  disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10#LAMBDA
    return gradient_penalty

class LossFake(nn.Module):
    def __init__(self):
        self.relu = nn.ReLU()

    def forward(self,x):
        loss = self.relu(x+1)
        return loss.mean()

class LossReal(nn.Module):
    def __init__(self):
        self.relu = nn.ReLU()

    def forward(self,x):
        loss = self.relu(x-1)
        return loss.mean()