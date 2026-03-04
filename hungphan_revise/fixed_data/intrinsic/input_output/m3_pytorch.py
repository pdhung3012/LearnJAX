import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

def train_test_loop(model, train_loader, test_loader, epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for image, label in train_loader:
            pred = model(image)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Training loss at epoch {epoch} = {loss.item()}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image_test, label_test in test_loader:
            pred_test = model(image_test)
            _, pred_test_vals = torch.max(pred_test, dim=1)
            total += label_test.size(0)
            correct += (pred_test_vals == label_test).sum().item()
    print(f"Test Accuracy = {(correct * 100)/total}")

class VanillaCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*16*16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def config_init(init_type="kaiming"):

    def kaiming_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)


    def xavier_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def zeros_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def random_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(m.weight)
            nn.init.normal_(m.bias)


    initializer_dict = {"kaiming": kaiming_init,
                        "xavier": xavier_init,
                        "zeros": zeros_init,
                        "random": random_init}

    return initializer_dict.get(init_type)

for name, model in zip(["Vanilla", "Kaiming", "Xavier", "Zeros", "Random"], [VanillaCNNModel(),
              VanillaCNNModel().apply(config_init("kaiming")),
              VanillaCNNModel().apply(config_init("xavier")),
              VanillaCNNModel().apply(config_init("zeros")),
              VanillaCNNModel().apply(config_init("random"))
              ]):
    print(f"_________{name}_______________________")
    train_test_loop(model, train_loader, test_loader)
