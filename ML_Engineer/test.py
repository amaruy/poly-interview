import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from train import SobelNet
import random
import cv2
import numpy as np

def load_trained_model(model_path):
    model = SobelNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Sobel filter function (same as before)
def apply_sobel(image):
    img = image.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min())
    return torch.from_numpy(sobel).unsqueeze(0).float()

def test_and_visualize():
    # Load the trained model
    model_path = "best_sobel_model.pth"
    trained_model = load_trained_model(model_path)

    # Load a random test image
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    random_index = random.randint(0, len(test_dataset) - 1)
    test_image, _ = test_dataset[random_index]

    # Randomly choose image size between 32 and 256
    random_size = random.randint(32, 256)
    resize_transform = transforms.Resize((random_size, random_size))
    test_image = resize_transform(test_image)

    # Apply the trained model
    with torch.no_grad():
        model_output = trained_model(test_image.unsqueeze(0)).squeeze(0)

    # Apply the actual Sobel filter
    sobel_output = apply_sobel(test_image)

    # Visualize the results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(test_image.permute(1, 2, 0))
    ax1.set_title(f"Original Image (Size: {random_size}x{random_size})")
    ax1.axis('off')

    ax2.imshow(model_output.squeeze().numpy(), cmap='gray')
    ax2.set_title("Model Output")
    ax2.axis('off')

    ax3.imshow(sobel_output.squeeze().numpy(), cmap='gray')
    ax3.set_title("Actual Sobel Filter")
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate Mean Squared Error
    mse = torch.mean((model_output - sobel_output) ** 2)
    print(f"Mean Squared Error between model output and actual Sobel filter: {mse.item():.6f}")

    print(f"Image size: {random_size}x{random_size}")
    print(f"Mean Squared Error between model output and actual Sobel filter: {mse.item():.6f}")

if __name__ == "__main__":
    test_and_visualize()
