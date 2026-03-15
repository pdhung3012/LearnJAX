import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Generate synthetic CT-scan data (batches, slices, RGB) and associated segmentation masks
torch.manual_seed(42)
batch = 100
num_slices = 10
channels = 3
width = 256
height = 256

ct_images = torch.randn(size=(batch, num_slices, channels, width, height))
segmentation_masks = (torch.randn(size=(batch, num_slices, 1, width, height))>0).float()

print(f"CT images (train examples) shape: {ct_images.shape}")
print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")

# Define the MedCNN class and its forward method
class MedCNN(nn.Module):
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
        self.relu = nn.ReLU()

    def forward(self, x):
        b, d, c, w, h = x.size()  #Input size: [B, D, C, W, H]
        print(f"Input shape [B, D, C, W, H]: {b, d, c, w, h}")

        x = x.view(b*d, c, w, h)  #Input to Resent 2DConv layers [B*D, C, W, H]
        features = self.backbone(x)
        print(f"ResNet output shape[B*D, C, W, H]: {features.shape}")

        _, new_c, new_w, new_h = features.size()
        x = features.view(b, d, new_c, new_w, new_h)  #[B, D, C, W, H]
        x = torch.permute(x, (0, 2, 1, 3, 4))  #rearrange for 3DConv layers [B, C, D, W, H]
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
        x = torch.sigmoid(self.final_conv(x))
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
    numerator = 2*torch.sum(pred*labels)
    denominator = torch.sum(pred) + torch.sum(labels) + eps
    return numerator/denominator

resnet_model = torchvision.models.resnet18(pretrained=True)
resnet_model = nn.Sequential(*list(resnet_model.children())[:-2])

model = MedCNN(backbone=resnet_model)

optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 5
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(ct_images)
    loss = compute_dice_loss(pred, segmentation_masks)
    loss.backward()
    optimizer.step()
    print(f"Loss at epoch {epoch}: {loss}")