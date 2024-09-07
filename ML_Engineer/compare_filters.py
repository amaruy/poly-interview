import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from train import SobelNet  

def load_trained_model(model_path):
    model = SobelNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_sobel_kernel():
    # Define the Sobel operator matrices
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Combine the matrices to get the Sobel kernel
    sobel_kernel = np.sqrt(sobel_x**2 + sobel_y**2)
    return sobel_kernel

def compare_filters():
    # Load the trained model
    model_path = "best_sobel_model.pth"
    trained_model = load_trained_model(model_path)
    
    # Get the trained filter
    trained_filter = trained_model.conv3.weight.detach().numpy()[0, 0]
    
    # Get the Sobel kernel
    sobel_kernel = get_sobel_kernel()
    
    # Normalize both filters
    trained_filter = (trained_filter - trained_filter.min()) / (trained_filter.max() - trained_filter.min())
    sobel_kernel = (sobel_kernel - sobel_kernel.min()) / (sobel_kernel.max() - sobel_kernel.min())
    
    # Plot the filters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(trained_filter, cmap='gray')
    ax1.set_title("Trained Filter")
    ax1.axis('off')
    
    ax2.imshow(sobel_kernel, cmap='gray')
    ax2.set_title("Sobel Kernel")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate Mean Squared Error
    mse = np.mean((trained_filter - sobel_kernel)**2)
    print(f"Mean Squared Error between trained filter and Sobel kernel: {mse:.6f}")

if __name__ == "__main__":
    compare_filters()
