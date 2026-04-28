import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_code_fixed_itr2 import (
    CHANNELS,
    EPOCHS,
    HEIGHT,
    LEARNING_RATE,
    NUM_SLICES,
    WIDTH,
    MedCNN,
    TruncatedResNet18Backbone,
    initialize_model_variables,
    train_step,
)


def load_fixture_data():
    ct_images = jnp.asarray(np.load("ct_images.npy"))
    segmentation_masks = jnp.asarray(np.load("segmentation_masks.npy"))
    return ct_images, segmentation_masks


def main():
    ct_images, segmentation_masks = load_fixture_data()
    print(f"CT images (train examples) shape: {ct_images.shape}")
    print(
        f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")

    model = MedCNN(backbone=TruncatedResNet18Backbone(), out_channel=1)
    rng_init = jax.random.PRNGKey(42)
    variables = initialize_model_variables(model, rng_init)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    print("Loaded pretrained ResNet18 weights into the JAX backbone.")

    print("\n" + "=" * 60)
    print("Model architecture (first forward pass):")
    print("=" * 60)
    dummy_input = jnp.ones(
        (1, NUM_SLICES, CHANNELS, WIDTH, HEIGHT), dtype=jnp.float32)
    _, updated_state = model.apply(
        {"params": params, "batch_stats": batch_stats},
        dummy_input,
        train=True,
        verbose=True,
        mutable=["batch_stats"],
    )
    batch_stats = updated_state["batch_stats"]

    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)

    print("\n" + "=" * 60)
    print("Training:")
    print("=" * 60)

    for epoch in range(EPOCHS):
        params, batch_stats, opt_state, loss = train_step(
            model,
            optimizer,
            params,
            batch_stats,
            opt_state,
            ct_images,
            segmentation_masks,
        )
        print(f"Loss at epoch {epoch}: {float(loss):.6f}")


if __name__ == "__main__":
    main()
