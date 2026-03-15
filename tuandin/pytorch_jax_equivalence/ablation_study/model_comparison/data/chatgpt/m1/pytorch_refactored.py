import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# -----------------------------
# Data utilities (importable)
# -----------------------------
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return torch.stack(in_seq), torch.stack(out_seq)


def generate_data():
    # Generate synthetic sequential data (same logic as original)
    sequence_length = 10
    num_samples = 100

    # Create a sine wave dataset
    X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
    y = torch.sin(X)

    # Prepare data for LSTM
    X_seq, y_seq = create_in_out_sequences(y, sequence_length)
    return X_seq, y_seq


# -----------------------------
# Models (module-level, importable)
# -----------------------------
class CustomLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        weights_biases_init = lambda: (
            nn.Parameter(torch.randn(input_dim, hidden_units)),
            nn.Parameter(torch.randn(hidden_units, hidden_units)),
            nn.Parameter(torch.zeros(hidden_units)),
        )
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.Wxi, self.Whi, self.bi = weights_biases_init()
        self.Wxf, self.Whf, self.bf = weights_biases_init()
        self.Wxo, self.Who, self.bo = weights_biases_init()
        self.Wxc, self.Whc, self.bc = weights_biases_init()
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, inputs, H_C=None):
        batch_size, seq_len, _ = inputs.shape
        if not H_C:
            H = torch.randn(batch_size, self.hidden_units)
            C = torch.randn(batch_size, self.hidden_units)
        else:
            H, C = H_C

        all_hidden_states = []
        for t in range(seq_len):
            X_t = inputs[:, t, :]
            I_t = torch.sigmoid(torch.matmul(X_t, self.Wxi) + torch.matmul(H, self.Whi) + self.bi)
            F_t = torch.sigmoid(torch.matmul(X_t, self.Wxf) + torch.matmul(H, self.Whf) + self.bf)
            O_t = torch.sigmoid(torch.matmul(X_t, self.Wxo) + torch.matmul(H, self.Who) + self.bo)
            C_tilde = torch.tanh(torch.matmul(X_t, self.Wxc) + torch.matmul(H, self.Whc) + self.bc)
            C = F_t * C + I_t * C_tilde
            H = O_t * torch.tanh(C)
            all_hidden_states.append(H.unsqueeze(1))

        outputs = torch.cat(all_hidden_states, dim=1)
        pred = self.fc(outputs)
        return pred, (H, C)


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# -----------------------------
# Factories (importable)
# -----------------------------
def make_model():
    model_custom = CustomLSTMModel(1, 50)
    model_inbuilt = LSTMModel()
    return model_custom, model_inbuilt


def make_criterion():
    return nn.MSELoss()


def make_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.01)


# -----------------------------
# Training (importable)
# -----------------------------
def train_model(X, y, model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        state = None
        pred, state = model(X, state)
        loss = criterion(pred[:, -1, :], y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def train_inbuilt_model(X, y, model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# -----------------------------
# Main (identical prints/flow)
# -----------------------------
def main():
    # Rule 8: seed at top of main, preserving original call order of RNG ops:
    # original RNG order was: manual_seed -> data torch.linspace (no RNG) -> sin (no RNG)
    # -> create_in_out_sequences (no RNG) -> model_custom init (torch.randn RNG)
    # -> model_inbuilt init (RNG) -> training forward where custom forward uses torch.randn for H,C each epoch
    torch.manual_seed(42)

    X_seq, y_seq = generate_data()

    model_custom, model_inbuilt = make_model()
    criterion = make_criterion()
    optimizer_custom = make_optimizer(model_custom)
    optimizer_inbuilt = make_optimizer(model_inbuilt)

    epochs = 500
    train_model(X_seq, y_seq, model_custom, optimizer_custom, criterion, epochs)

    epochs = 500
    train_inbuilt_model(X_seq, y_seq, model_inbuilt, optimizer_inbuilt, criterion, epochs)

    # Testing on new data (same logic as original)
    sequence_length = 10
    test_steps = 100
    X_test = torch.linspace(0, 5 * 3.14159, steps=test_steps).unsqueeze(1)
    y_test = torch.sin(X_test)

    X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

    with torch.no_grad():
        pred_custom, _ = model_custom(X_test_seq)
        pred_inbuilt = model_inbuilt(X_test_seq)

    pred_custom = torch.flatten(pred_custom[:, -1, :])
    pred_inbuilt = pred_inbuilt.squeeze()
    print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
    print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")

    # Plot the predictions (same behavior)
    plt.figure()
    plt.plot(pred_custom, label="custom model")
    plt.plot(pred_inbuilt, label="inbuilt model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()