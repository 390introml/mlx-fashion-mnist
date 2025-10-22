# Fashion MNIST pipeline in PyTorch
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.decomposition import PCA

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fashion MNIST class names
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load Fashion MNIST dataset
def load_fashion_mnist(batch_size=64):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])
    
    # Download and load the dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, 
                                        download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, 
                                       download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Define the neural network model
class FashionMNISTNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256):
        super(FashionMNISTNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 512)
        self.fc4 = nn.Linear(512, input_size)
        self.relu = nn.ReLU()
    
    def encoder(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def decoder(self, x):
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Loss function (MSE for autoencoder)
def loss_fn(output, target):
    return nn.MSELoss()(output, target)

# Training function
def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = loss_fn(output, data)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--representation_size", type=int, default=256, help="Hidden layer size for the neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # Update hyperparameters from command line arguments
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate

    print(f"Using device: {device}")
    print("Loading Fashion MNIST dataset...")
    train_loader, test_loader = load_fashion_mnist(BATCH_SIZE)

    print("Initializing model...")
    model = FashionMNISTNet(hidden_size=args.representation_size).to(device)

    print("Starting training...")
    train_model(model, train_loader, EPOCHS, LEARNING_RATE)

    print("Training completed!")

    # Visualizations
    print("\nGenerating visualizations...")
    model.eval()

    # Get test data for visualization
    test_data = []
    test_labels = []
    with torch.no_grad():
        for data, labels in test_loader:
            test_data.append(data)
            test_labels.append(labels)
            if len(test_data) >= 1:  # Just get first batch
                break
    
    test_images = test_data[0].to(device)
    test_labels = test_labels[0]

    # 1. PCA visualization of encoded representations
    print("Creating PCA visualization of encoded representations...")
    num_samples = 20
    sample_images = test_images[:num_samples]
    sample_labels = test_labels[:num_samples]

    # Encode the samples
    with torch.no_grad():
        encoded = model.encoder(sample_images)
    encoded_np = encoded.cpu().numpy()

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    encoded_2d = pca.fit_transform(encoded_np)

    # Create the plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1],
                         c=sample_labels.numpy(), cmap='tab10', s=100)

    # Annotate each point with its label
    for i, (x, y) in enumerate(encoded_2d):
        label_idx = int(sample_labels[i])
        plt.annotate(CLASS_NAMES[label_idx],
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7)

    plt.colorbar(scatter, label='Class')
    plt.title(f'PCA of Encoded Representations (first {num_samples} test samples)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_encoded.png', dpi=150)
    print("Saved PCA visualization to pca_encoded.png")

    # 2 & 3. Random sample and reconstruction
    print("Creating original and reconstructed image comparison...")
    random_idx = np.random.randint(0, test_images.shape[0])
    sample_image = test_images[random_idx:random_idx+1]
    sample_label = int(test_labels[random_idx])

    # Get reconstruction
    with torch.no_grad():
        reconstructed = model(sample_image)

    # Reshape for visualization (28x28)
    original_img = sample_image[0].cpu().numpy().reshape(28, 28)
    reconstructed_img = reconstructed[0].cpu().numpy().reshape(28, 28)

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'Original Image\n({CLASS_NAMES[sample_label]})')
    axes[0].axis('off')

    axes[1].imshow(reconstructed_img, cmap='gray')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=150)
    print("Saved reconstruction comparison to reconstruction_comparison.png")

    print("\nAll visualizations completed!")