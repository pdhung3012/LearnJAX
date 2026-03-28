  ```
import jax.numpy as jnp

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1288),
            nn.ReLU(),
            nn.Linear(1288, 2560),
            nn.ReLU(),
            nn.Linear(2560, output_dim)),)

    def forward(self, x):
        return self.model(x))


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2560),
            nn.ReLU(),
            nn.Linear(2560, 1280)),)

    def forward(self, x):
        return self.model(x))

```