import torch
import torch.nn as nn
import torch.optim as optim

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Generate synthetic data for training
torch.manual_seed(42)
real_data = torch.rand(100, 1) * 2 - 1  # 100 samples in the range [-1, 1]

# Initialize models, loss, and optimizers
latent_dim = 10
data_dim = 1
G = Generator(latent_dim, data_dim)
D = Discriminator(data_dim)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.001)
optimizer_D = optim.Adam(D.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Train Discriminator
    latent_samples = torch.randn(real_data.size(0), latent_dim)
    fake_data = G(latent_samples).detach()
    real_labels = torch.ones(real_data.size(0), 1)
    fake_labels = torch.zeros(real_data.size(0), 1)

    optimizer_D.zero_grad()
    real_loss = criterion(D(real_data), real_labels)
    fake_loss = criterion(D(fake_data), fake_labels)
    loss_D = real_loss + fake_loss
    loss_D.backward()
    optimizer_D.step()

    # Train Generator
    latent_samples = torch.randn(real_data.size(0), latent_dim)
    fake_data = G(latent_samples)
    optimizer_G.zero_grad()
    loss_G = criterion(D(fake_data), real_labels)
    loss_G.backward()
    optimizer_G.step()

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

# Generate new samples with the trained Generator
latent_samples = torch.randn(5, latent_dim)
with torch.no_grad():
    generated_data = G(latent_samples)
    print(f"Generated data: {generated_data.tolist()}")