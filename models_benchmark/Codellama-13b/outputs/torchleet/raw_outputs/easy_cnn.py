  ```
import jax
import jax.numpy as np
# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2))  # Output: 64x16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 128))  # Output: 128
        self.fc2 = nn.Linear(128, 10)))  # Output: 10

    def forward(self, x):
        x = np.maximum(0, x))))  # ReLU

        x = self.conv1(x))  # Conv layer

        x = self.pool(np.maximum(0, x)))))))  # Pooling layer

        x = self.fc1(x))  # Fully connected layer



        x = self.fc2(x))  # Fully connected layer










```