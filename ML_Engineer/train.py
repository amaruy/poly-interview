import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import cv2
import numpy as np

# Sobel filter function (same as before)
def apply_sobel(image):
    img = image.permute(1, 2, 0).numpy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min())
    return torch.from_numpy(sobel).unsqueeze(0).float()

# Custom Dataset
class SobelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        sobel_image = apply_sobel(image)
        return image, sobel_image

class SobelNet(nn.Module):
    def __init__(self):
        super(SobelNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.conv4(x)

if __name__ == "__main__":
    # Data loading
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    sobel_dataset = SobelDataset(cifar_dataset)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(sobel_dataset))
    val_size = len(sobel_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(sobel_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SobelNet().to(device)

    # # Initialize weights to resemble Sobel filter
    # with torch.no_grad():
    #     sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    #     sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    #     sobel_filter = torch.sqrt(sobel_x.pow(2) + sobel_y.pow(2)).unsqueeze(0).unsqueeze(0)
    #     sobel_filter = sobel_filter.repeat(1, 3, 1, 1) / 3  # Repeat for 3 input channels
    #     model.conv.weight.data.copy_(sobel_filter)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    patience = 15
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (images, sobel_images) in enumerate(train_loader):
            images, sobel_images = images.to(device), sobel_images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, sobel_images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, sobel_images in val_loader:
                images, sobel_images = images.to(device), sobel_images.to(device)
                outputs = model(images)
                loss = criterion(outputs, sobel_images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_sobel_model.pth")
        else:
            no_improve += 1
        
        if no_improve == patience:
            print("Early stopping!")
            break

    print("Training finished!")

    # Load best model
    model.load_state_dict(torch.load("best_sobel_model.pth"))