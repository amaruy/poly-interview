# Sobel Filter Neural Network

This project implements a neural network that learns to apply a Sobel filter to images. The network is trained on a dataset of input images and their corresponding Sobel-filtered outputs.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/amaruy/poly-interview.git
   cd poly-interview/ML_Engineer
   ```

2. Install the required packages:
   ```
   pip install torch torchvision matplotlib opencv-python numpy
   ```

## Usage

### 1. Visualize Sobel Operator

To see how the Sobel operator works on a sample image:

```
python sobel_operator.py
```

This script will display the original image and the Sobel-filtered version side by side.

### 2. Train the Model

To train the neural network:

```
python train.py
```

This script will:
- Load the CIFAR10 dataset
- Create a custom SobelDataset
- Train the SobelNet model
- Save the best model as `best_sobel_model.pth`

### 3. Test the Model

To test the trained model on a random image:

```
python test.py
```

This script will:
- Load a random test image from CIFAR10
- Resize it to a random size between 32x32 and 256x256
- Apply the trained model and the actual Sobel filter
- Display the results and calculate the Mean Squared Error

### 4. Compare Filters

To compare the learned filter with the actual Sobel kernel:

```
python compare_filters.py
```

This script will visualize the learned filter and the Sobel kernel side by side, and calculate the Mean Squared Error between them.

## Design Considerations

1. **Network Architecture**: A simple 3-layer convolutional neural network (SobelNet) is used. This lightweight architecture is sufficient for learning the Sobel filter operation.

2. **Variable Image Sizes**: The SobelDataset class supports variable image sizes, allowing the model to generalize to different input dimensions.

3. **Reflection Padding**: The convolutional layers use reflection padding to handle edge cases, ensuring consistent output sizes.

4. **Early Stopping**: The training process implements early stopping to prevent overfitting and reduce unnecessary computation.

5. **Learning Rate Scheduling**: A learning rate scheduler is used to adjust the learning rate during training, helping to fine-tune the model's performance.

6. **Model Compilation**: The trained model is compiled using torch.jit.script for faster inference.

7. **Visualization**: Various scripts are provided to visualize the Sobel operator, test results, and compare the learned filter with the actual Sobel kernel.

These design choices aim to create a flexible, efficient, and interpretable implementation of a Sobel filter neural network.

