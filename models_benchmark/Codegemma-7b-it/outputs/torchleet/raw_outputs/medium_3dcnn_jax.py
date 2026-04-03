import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Generate synthetic CT-scan data (batches, slices, RGB) and associated segmentation masks
batch = 100
num_slices = 10
channels = 3
width = 256
height = 256

ct_images = jnp.random.randn(size=(batch, num_slices, channels, width, height))
segmentation_masks = (jnp.random.randn(size=(batch, num_slices, 1, width, height))>0).astype(jnp.float32)

print(f"CT images (train examples) shape: {ct_images.shape}")
print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")

# Define the MedCNN class and its forward method
class MedCNN(nn.Module):
    @nn.compact
    def __init__(self, backbone, out_channel=1):
        super(MedCNN, self).__init__()
        self.backbone = backbone
        
        #Downsample
        self.conv1 = nn.Conv3d(512, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=1)
        
        #Upsample
        self.conv_transpose1 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.conv_transpose2 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 8, 8), stride=(1, 8, 8))
        
        #Final convolution layer from 16 to 1 channel
        self.final_conv = nn.Conv3d(16, out_channel, kernel_size=1)
        self.relu = nn.relu

    def call(self, x):
        b, d, c, w, h = x.shape #Input size: [B, D, C, W, H]
        print(f"Input shape [B, D, C, W, H]: {b, d, c, w, h}")
        
        x = x.reshape(b*d, c, w, h) #Input to Resent 2DConv layers [B*D, C, W, H]
        features = self.backbone(x)
        print(f"ResNet output shape[B*D, C, W, H]: {features.shape}")
        
        _, new_c, new_w, new_h = features.shape
        x = features.reshape(b, d, new_c, new_w, new_h) #[B, D, C, W, H]
        x = jnp.transpose(x, (0, 2, 1, 3, 4)) #rearrange for 3DConv layers [B, C, D, W, H]
        print(f"Reshape Resnet output for 3DConv #1 [B, C, D, W, H]: {x.shape}")
        
        #Downsampling
        x = self.relu(self.conv1(x))
        print(f"Output shape 3D Conv #1: {x.shape}")
        x = self.relu(self.conv2(x))
        print(f"Output shape 3D Conv #2: {x.shape}")
        
        #Upsampling
        x = self.relu(self.conv_transpose1(x))
        print(f"Output shape 3D Transposed Conv #1: {x.shape}")
        x = self.relu(self.conv_transpose2(x))
        print(f"Output shape 3D Transposed Conv #2: {x.shape}")

        #final segmentation
        x = jnp.sigmoid(self.final_conv(x))
        print(f"Final shape: {x.shape}")
        
        return x

def compute_dice_loss(pred, labels, eps=1e-8):
    '''
    Args
    pred: [B, D, 1, W, H]
    labels: [B, D, 1, W, H]
    
    Returns
    dice_loss: [B, D, 1, W, H]
    '''
    numerator = 2*jnp.sum(pred*labels)
    denominator = jnp.sum(pred) + jnp.sum(labels) + eps
    return numerator/denominator

resnet_model = jax.nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-2])

model = MedCNN(backbone=resnet_model)

optimizer = optax.adam(learning_rate=0.01)

@jax.jit
def train_step(params, batch):
    ct_images, segmentation_masks = batch
    def loss_fn(params):
        pred = model.apply({'params': params}, ct_images)
        loss = compute_dice_loss(pred, segmentation_masks)
        return loss, pred
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, pred), grads = grad_fn(params)
    params = optax.apply_updates(params, grads)
    return loss, params

epochs = 5
for epoch in range(epochs):
    loss, params = train_step(params, (ct_images, segmentation_masks))
    print(f"Loss at epoch {epoch}: {loss}")