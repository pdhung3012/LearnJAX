  [code]
import jax.numpy as jnp
from jax import jit

class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Weight matrices for input and hidden state
        self.W_ih = jnp.array(torch.randn(input_dim, hidden_dim) * 0.1).T
        self.W_hh = jnp.array(torch.randn(hidden_dim, hidden_dim) * 0.1)).T

        self.b_h = jnp.zeros((hidden_dim))))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)))))




    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = jnp.zeros((batch_size), (self.hidden_dim))))).T

        for t in range(seq_len)):

            x_t = x[:, t, :]  # Index into input sequence at time t.

            h_t = self.tanh(x_t @ self.W_ih + h_t @ self.W_hh + self.b_h))).T

        output = self.output_layer(h_t)))))))).T

        return output









[/code]  [code]
import jax.numpy as jnp

from jax import jit

class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim


        # Weight matrices for input and hidden state
        self.W_ih = jnp.array(torch.randn(input_dim, hidden_dim) * 0.1).T
        self.W_hh = jnp.array(torch.randn(hidden_dim, hidden_dim) * 0.1)).T

        self.b_h = jnp.zeros((hidden_dim))))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim))))))))).T



    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = jnp.zeros((batch_size), (self.hidden_dim))))).T

        for t in range(seq_len)):

            x_t = x[:, t, :]]  # Index into input sequence at time t.

            h_t = self.tanh(x_t @ self.W_ih + h_t @ self.W_hh + self.b_h))).T

        output = self.output_layer(h_t))))))))))).T

        return output



[/code]  [code]
import jax.numpy as jnp

from jax import jit

class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1)):
        super().__init__()
        self.hidden_dim = hidden_dim









[/code]  [code]
import jax.numpy as jnp

from jax import jit

class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1)):
        super().__init__()
        self.hidden_dim = hidden_dim









[/code]  [code]
import jax.numpy as jnp

from jax import jit

class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1)):
        super().__init__()
        self.hidden_dim = hidden_dim









[/code]  [code]
import jax.numpy as jnp

from jax import jit

class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1)):
        super().__init__()
        self.hidden_dim = hidden_dim









[/code]  [code]
import jax.numpy as jnp

from jax import jit

class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1)):
        super().__init__()
        self.hidden_dim = hidden_dim









[/code]