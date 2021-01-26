
class loss_discriminator(nn.Module):
    def __init__(self):
        self.criterion = nn.L1Loss()
