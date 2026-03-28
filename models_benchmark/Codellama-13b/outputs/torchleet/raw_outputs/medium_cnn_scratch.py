  ```
import jax.numpy as jnp
from flax import linen as nn
class Conv2dCustom(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0)):
        super(Conv2dCustom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


    def forward(self, x)));



class MaxPool2dCustom(nn.Module):

    def __init__(self, kernel_size, stride=None)):

        super(MaxPool2dCustom, self).__init__()

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)




class CNNModel(nn.Module):
    def __init__(self));


        super(CNNModel, self).__init__()


        self.conv1 = Conv2dCustom(3, 32, kernel_size=3, stride=1, padding=1))








        self.pool = MaxPool2dCustom(kernel_size=2, stride=2)))








        self.fc1 = LinearCustom(64 * 16 * 16, 128))








        self.fc2 = LinearCustom(128, 10))








        self.relu = ReLU())









    def forward(self, x)));





