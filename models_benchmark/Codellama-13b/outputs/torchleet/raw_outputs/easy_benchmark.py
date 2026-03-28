  [code]



# Define a simple neural network model

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Initialize the first fully connected layer with 28x28 pixels as input and 128 neurons as output
        self.fc1 = nn.Linear(28*28, 128)

    def forward(self, x):
        # Forward pass through the first fully connected layer

        x = torch.relu(self.fc1(x))))


        return x



# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)











