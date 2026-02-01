import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

# Define the Decoder with Attention
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, src_seq_length):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.attention = nn.Linear(hidden_dim + embed_dim, src_seq_length)
        self.attention_combine = nn.Linear(hidden_dim + embed_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, encoder_outputs, hidden, cell):
        x = x.unsqueeze(1)  # Add sequence dimension
        embedded = self.embedding(x)

        # Attention mechanism
        attention_weights = torch.softmax(self.attention(torch.cat((embedded.squeeze(1), hidden[-1]), dim=1)), dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)

        # Combine context and embedded input
        combined = torch.cat((embedded.squeeze(1), context_vector.squeeze(1)), dim=1)
        combined = torch.tanh(self.attention_combine(combined)).unsqueeze(1)

        # LSTM and output
        lstm_out, (hidden, cell) = self.lstm(combined, (hidden, cell))
        output = self.fc_out(lstm_out.squeeze(1))
        return output, hidden, cell

# Define synthetic training data
torch.manual_seed(42)
src_vocab_size = 20
tgt_vocab_size = 20
src_seq_length = 10
tgt_seq_length = 12
batch_size = 16

src_data = torch.randint(0, src_vocab_size, (batch_size, src_seq_length))
tgt_data = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length))

# Initialize models, loss function, and optimizer
input_dim = src_vocab_size
output_dim = tgt_vocab_size
embed_dim = 32
hidden_dim = 64
num_layers = 2

encoder = Encoder(input_dim, embed_dim, hidden_dim, num_layers)
decoder = Decoder(output_dim, embed_dim, hidden_dim, num_layers, src_seq_length)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    encoder_outputs, (hidden, cell) = encoder(src_data)
    loss = 0
    decoder_input = torch.zeros(batch_size, dtype=torch.long)  # Start token

    for t in range(tgt_seq_length):
        output, hidden, cell = decoder(decoder_input, encoder_outputs, hidden, cell)
        loss += criterion(output, tgt_data[:, t])
        decoder_input = tgt_data[:, t]  # Teacher forcing

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

# Test the sequence-to-sequence model with new input
test_input = torch.randint(0, src_vocab_size, (1, src_seq_length))
with torch.no_grad():
    encoder_outputs, (hidden, cell) = encoder(test_input)
    decoder_input = torch.zeros(1, dtype=torch.long)  # Start token
    output_sequence = []

    for _ in range(tgt_seq_length):
        output, hidden, cell = decoder(decoder_input, encoder_outputs, hidden, cell)
        predicted = output.argmax(1)
        output_sequence.append(predicted.item())
        decoder_input = predicted

    print(f"Input: {test_input.tolist()}, Output: {output_sequence}")
