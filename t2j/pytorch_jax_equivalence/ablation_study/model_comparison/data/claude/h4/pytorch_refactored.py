import torch
import torch.nn as nn
import torch.optim as optim


# Define the Generator (module-level, importable)
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


# Define the Discriminator (module-level, importable)
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


def generate_data():
    """Generate synthetic data for training: 100 samples in [-1, 1]."""
    real_data = torch.rand(100, 1) * 2 - 1
    # y is not used in the original GAN training, but we return a placeholder
    # to satisfy the (X, y) return signature required by the rules.
    y = None
    return real_data, y


def make_model(latent_dim=10, data_dim=1):
    """Instantiate and return the Generator and Discriminator.

    Weight initialisation is unchanged — uses PyTorch defaults.
    """
    G = Generator(latent_dim, data_dim)
    D = Discriminator(data_dim)
    return G, D


def make_criterion():
    """Return the loss function."""
    return nn.BCELoss()


def make_optimizer(model, lr=0.001):
    """Return an Adam optimizer with the same hyperparameters as the original."""
    return optim.Adam(model.parameters(), lr=lr)


def train_model(X, y, G, D, optimizer_G, optimizer_D, criterion, num_epochs):
    """Training loop with identical logic to the original script.

    Parameters
    ----------
    X : Tensor
        Real data (referred to as real_data in the original).
    y : ignored
        Unused; kept for interface consistency.
    G : Generator
    D : Discriminator
    optimizer_G : optimizer for the Generator
    optimizer_D : optimizer for the Discriminator
    criterion : loss function
    num_epochs : int
    """
    real_data = X
    latent_dim = G.model[0].in_features  # recover latent_dim from Generator

    for epoch in range(num_epochs):
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
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")


def main():
    # Seed set here, matching the original module-level position before any
    # random operations. Call order preserved: seed -> generate_data -> make_model
    # -> make_criterion -> make_optimizer (G then D) -> train -> generate samples.
    torch.manual_seed(42)

    latent_dim = 10
    data_dim = 1
    epochs = 1000

    X, y = generate_data()
    G, D = make_model(latent_dim, data_dim)
    criterion = make_criterion()
    optimizer_G = make_optimizer(G, lr=0.001)
    optimizer_D = make_optimizer(D, lr=0.001)

    train_model(X, y, G, D, optimizer_G, optimizer_D, criterion, epochs)

    # Generate new samples with the trained Generator
    latent_samples = torch.randn(5, latent_dim)
    with torch.no_grad():
        generated_data = G(latent_samples)
        print(f"Generated data: {generated_data.tolist()}")


if __name__ == '__main__':
    main()