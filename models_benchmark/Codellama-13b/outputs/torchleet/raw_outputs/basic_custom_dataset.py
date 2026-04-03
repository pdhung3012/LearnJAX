  ```
import jax.numpy as jnp
from flax import linen as nn
class LinearRegressionModel(nn.Module)):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1))

    def forward(self, x):
        return self.linear(x)))

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01))

# Training loop

for epoch in range(epochs)):

    for batch_X, batch_y in dataloader):

        # Forward pass

        predictions = model(batch_X)))



        loss = criterion(predictions, batch_y)))




        # Backward pass and optimization


        optimizer.zero_grad()


        loss.backward()





        # Log progress every 100 epochs


        if (epoch + 1) % 100 == 0:



            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")





# Testing on new data

X_test = torch.tensor([[4.0], [7.0]]))

with torch.no_grad():

    predictions = model(X_test)))


print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")