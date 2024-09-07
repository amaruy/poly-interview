import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from train import SobelNet, apply_sobel

def load_trained_model(model_path):
    model = SobelNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test_and_visualize():
    # Load the trained model
    model_path = "best_sobel_model.pth"
    trained_model = load_trained_model(model_path)

    # Load a test image
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_image, _ = test_dataset[0]

    # Apply the trained model
    with torch.no_grad():
        model_output = trained_model(test_image.unsqueeze(0)).squeeze(0)

    # Apply the actual Sobel filter
    sobel_output = apply_sobel(test_image)

    # Visualize the results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(test_image.permute(1, 2, 0))
    ax1.set_title("Original Image")
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

if __name__ == "__main__":
    test_and_visualize()
