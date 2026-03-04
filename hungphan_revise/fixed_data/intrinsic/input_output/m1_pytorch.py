## Original Code from TorchLeet ##
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic sequential data
torch.manual_seed(42)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
y = torch.sin(X)

# Prepare data for LSTM
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return torch.stack(in_seq), torch.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

class CustomLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        weights_biases_init = lambda : (nn.Parameter(torch.randn(input_dim, hidden_units)),
                                        nn.Parameter(torch.randn(hidden_units, hidden_units)),
                                        nn.Parameter(torch.zeros(hidden_units)))
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.Wxi, self.Whi, self.bi = weights_biases_init()
        self.Wxf, self.Whf, self.bf = weights_biases_init()
        self.Wxo, self.Who, self.bo = weights_biases_init()
        self.Wxc, self.Whc, self.bc = weights_biases_init()
        self.fc = nn.Linear(hidden_units, 1)
        # print(self.Wxi.shape, self.Whi.shape, self.bi.shape)

    def forward(self, inputs, H_C=None):
        # print(inputs.shape, self.Wxi.shape)
        batch_size, seq_len, _ = inputs.shape
        if not H_C:
            H = torch.randn(batch_size, self.hidden_units)
            C = torch.randn(batch_size, self.hidden_units)
        else:
            H, C = H_C

        all_hidden_states = []
        for t in range(seq_len):
            X_t = inputs[:, t, :]
            # print(X.shape, self.Wxi.shape, self.Whi.shape, self.bi.shape)
            I_t = torch.sigmoid(torch.matmul(X_t, self.Wxi) + torch.matmul(H, self.Whi) + self.bi)
            F_t = torch.sigmoid(torch.matmul(X_t, self.Wxf) + torch.matmul(H, self.Whf) + self.bf)
            O_t = torch.sigmoid(torch.matmul(X_t, self.Wxo) + torch.matmul(H, self.Who) + self.bo)
            C_tilde = torch.tanh(torch.matmul(X_t, self.Wxc) + torch.matmul(H, self.Whc) + self.bc)
            C = F_t * C + I_t * C_tilde
            H = O_t * torch.tanh(C)
            # print(H.shape)
            all_hidden_states.append(H.unsqueeze(1))

        outputs = torch.cat(all_hidden_states, dim=1)
        pred = self.fc(outputs)
        # print(pred.shape)
        return pred, (H, C)

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the LSTM
        return out

# Initialize the model, loss function, and optimizer
model_custom = CustomLSTMModel(1, 50)
model_inbuilt = LSTMModel()
criterion = nn.MSELoss()
optimizer_custom = optim.Adam(model_custom.parameters(), lr=0.01)
optimizer_inbuilt = optim.Adam(model_inbuilt.parameters(), lr=0.01)

# Training loop for the custom model
epochs = 500
for epoch in range(epochs):
    # Forward pass
    state = None
    pred, state = model_custom(X_seq, state)
    loss = criterion(pred[:, -1, :], y_seq) # Use the last output of the LSTM
    # Backward pass and optimization
    optimizer_custom.zero_grad()
    loss.backward()
    optimizer_custom.step()

    # Log progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Training loop for the inbuilt model
epochs = 500
for epoch in range(epochs):
    # Forward pass
    pred = model_inbuilt(X_seq)
    loss = criterion(pred, y_seq)
    # Backward pass and optimization
    optimizer_inbuilt.zero_grad()
    loss.backward()
    optimizer_inbuilt.step()

    # Log progress every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testing on new data
test_steps = 100  # Ensure this is greater than sequence_length
X_test = torch.linspace(0, 5 * 3.14159, steps=test_steps).unsqueeze(1)
y_test = torch.sin(X_test)

# Create test input sequences
X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

with torch.no_grad():
    pred_custom, _ = model_custom(X_test_seq)
    pred_inbuilt = model_inbuilt(X_test_seq)
pred_custom = torch.flatten(pred_custom[:, -1, :])
pred_inbuilt = pred_inbuilt.squeeze()
print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")

#Plot the predictions
plt.figure()
# plt.plot(y_test, label="Ground Truth")
plt.plot(pred_custom, label="custom model")
plt.plot(pred_inbuilt, label="inbuilt model")
plt.legend()
plt.show()
