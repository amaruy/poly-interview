import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel(image):
    # Convert PyTorch tensor to numpy array
    img = image.permute(1, 2, 0).numpy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Sobel filter
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min())
    
    # Convert back to PyTorch tensor (single channel)
    return torch.from_numpy(sobel).unsqueeze(0).float()

# Load CIFAR10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Get a sample image
image, _ = dataset[0]

# Apply Sobel filter
sobel_image = apply_sobel(image)

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image.permute(1, 2, 0))
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(sobel_image[0].cpu().numpy(), cmap='gray')  # Use the first (and only) channel
ax2.set_title('Sobel Filter Applied')
ax2.axis('off')
plt.show()

# Print shapes
print("Original image shape:", image.shape)
print("Sobel image shape:", sobel_image.shape)
