import torch
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad

# Import your actual components
from pytorch_refactored import SimpleModel, make_model, make_criterion, make_optimizer
from jax_code_fixed import simple_model as jax_model, make_criterion as jax_make_criterion

def debug_comparison():
    # 1. Setup identical hardcoded data
    X_np = np.linspace(0, 1, 10).reshape(10, 1).astype(np.float32)
    y_np = (X_np * 3.0 + 2.0).astype(np.float32)
    W_init = np.array([[0.5]], dtype=np.float32)
    B_init = np.array([0.1], dtype=np.float32)

    # 2. ANALYTICAL TRUTH (Manual Math)
    # Loss = mean((W*X + B - y)^2)
    # dL/dW = mean(2 * (W*X + B - y) * X)
    preds_ana = X_np @ W_init.T + B_init
    error = preds_ana - y_np
    grad_W_ana = np.mean(2 * error * X_np)
    grad_B_ana = np.mean(2 * error)

    # 3. PYTORCH GRADIENT
    pt_model = make_model()
    with torch.no_grad():
        pt_model.fc.weight.data = torch.from_numpy(W_init)
        pt_model.fc.bias.data = torch.from_numpy(B_init)
    
    pt_criterion = make_criterion()
    pt_preds = pt_model(torch.from_numpy(X_np))
    pt_loss = pt_criterion(pt_preds, torch.from_numpy(y_np))
    pt_loss.backward()
    
    grad_W_pt = pt_model.fc.weight.grad.numpy()
    grad_B_pt = pt_model.fc.bias.grad.numpy()

    # 4. JAX GRADIENT
    jax_params = {'w': jnp.array(W_init).T, 'b': jnp.array(B_init)}
    jax_loss_fn = jax_make_criterion()
    loss_jax, grads_jax = value_and_grad(jax_loss_fn)(jax_params, jnp.array(X_np), jnp.array(y_np))
    
    grad_W_jax = np.array(grads_jax['w']).T
    grad_B_jax = np.array(grads_jax['b'])

    print(f"{'Metric':<20} | {'Analytical':<15} | {'PyTorch':<15} | {'JAX':<15}")
    print("-" * 70)
    print(f"{'Loss':<20} | {np.mean(error**2):<15.6f} | {pt_loss.item():<15.6f} | {loss_jax:<15.6f}")
    print(f"{'Grad W':<20} | {grad_W_ana:<15.6f} | {grad_W_pt[0,0]:<15.6f} | {grad_W_jax[0,0]:<15.6f}")
    print(f"{'Grad B':<20} | {grad_B_ana:<15.6f} | {grad_B_pt[0]:<15.6f} | {grad_B_jax[0]:<15.6f}")

    # 5. Check the Update
    lr = 0.01
    print("\n--- Predicted Weight after 1 step (lr=0.01) ---")
    print(f"PyTorch W: {W_init[0,0] - lr * grad_W_pt[0,0]:.6f}")
    print(f"JAX W:     {W_init[0,0] - lr * grad_W_jax[0,0]:.6f}")

if __name__ == "__main__":
    debug_comparison()