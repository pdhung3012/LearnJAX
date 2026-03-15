import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Module Level Classes ---

class CustomLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super().__init__()
        weights_biases_init = lambda : (
            nn.Parameter(torch.randn(input_dim, hidden_units)), 
            nn.Parameter(torch.randn(hidden_units, hidden_units)), 
            nn.Parameter(torch.zeros(hidden_units))
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

# --- Helper Components ---

def generate_data():
    sequence_length = 10
    num_samples = 100
    X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
    y = torch.sin(X)

    def create_in_out_sequences(data, seq_length):
        in_seq = []
        out_seq = []
        for i in range(len(data) - seq_length):
            in_seq.append(data[i:i + seq_length])
            out_seq.append(data[i + seq_length])
        return torch.stack(in_seq), torch.stack(out_seq)

    X_seq, y_seq = create_in_out_sequences(y, sequence_length)
    return X_seq, y_seq

def make_model(model_type="custom"):
    if model_type == "custom":
        return CustomLSTMModel(1, 50)
    else:
        return LSTMModel()

def make_criterion():
    return nn.MSELoss()

def make_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.01)

def train_model(X, y, model, optimizer, criterion, num_epochs, is_custom=True):
    for epoch in range(num_epochs):
        if is_custom:
            state = None
            pred, state = model(X, state)
            loss = criterion(pred[:, -1, :], y)
        else:
            pred = model(X)
            loss = criterion(pred, y)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# --- Main Script Execution ---

def main():
    # 1. Set seed first
    torch.manual_seed(42)

    # 2. Generate Data (matches original order)
    X_seq, y_seq = generate_data()

    # 3. Instantiate models (Custom then Inbuilt to preserve RNG state for weight init)
    model_custom = make_model("custom")
    model_inbuilt = make_model("inbuilt")

    # 4. Setup criteria and optimizers
    criterion = make_criterion()
    optimizer_custom = make_optimizer(model_custom)
    optimizer_inbuilt = make_optimizer(model_inbuilt)

    # 5. Train Custom Model
    train_model(X_seq, y_seq, model_custom, optimizer_custom, criterion, 500, is_custom=True)

    # 6. Train Inbuilt Model
    train_model(X_seq, y_seq, model_inbuilt, optimizer_inbuilt, criterion, 500, is_custom=False)

    # 7. Testing / Inference
    test_steps = 100
    X_test = torch.linspace(0, 5 * 3.14159, steps=test_steps).unsqueeze(1)
    y_test = torch.sin(X_test)
    
    # Re-using the sequence logic from generate_data context
    def create_sequences(data, seq_length=10):
        in_seq = []
        for i in range(len(data) - seq_length):
            in_seq.append(data[i:i + seq_length])
        return torch.stack(in_seq)

    X_test_seq = create_sequences(y_test)

    with torch.no_grad():
        pred_custom, _ = model_custom(X_test_seq)
        pred_inbuilt = model_inbuilt(X_test_seq)

    pred_custom_final = torch.flatten(pred_custom[:, -1, :])
    pred_inbuilt_final = pred_inbuilt.squeeze()

    print(f"Predictions with Custom Model for new sequence: {pred_custom_final.tolist()}")
    print(f"Predictions with In-Built Model: {pred_inbuilt_final.tolist()}")

    # 8. Plotting
    plt.figure()
    plt.plot(pred_custom_final, label="custom model")
    plt.plot(pred_inbuilt_final, label="inbuilt model")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()