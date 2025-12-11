# MNIST Classification Lab: CNN, Faster R-CNN, Pretrained Models, and Vision Transformer #
##########################################################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import vgg16, alexnet
import torchvision.models.detection as detection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import time
from PIL import Image
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Loading and Preprocessing

def load_mnist_idx(images_path, labels_path):
    """Load MNIST data from IDX format files"""
    # Load images
    with open(images_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    
    # Load labels
    with open(labels_path, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return images, labels

class MNISTDataset(Dataset):
    """Custom MNIST Dataset loader for IDX format"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data transformations
transform_cnn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_pretrained = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),  # Convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# First, let's explore the dataset structure
print("Exploring dataset structure...")
import glob

base_path = '/kaggle/input/mnist-dataset'
all_files = []
for ext in ['**/*', '*']:
    all_files.extend(glob.glob(os.path.join(base_path, ext), recursive=True))

print("\nAvailable files:")
for f in sorted(all_files):
    if os.path.isfile(f):
        print(f"  FILE: {f}")
    else:
        print(f"  DIR:  {f}")

# Find the actual data files
train_images_path = None
train_labels_path = None
test_images_path = None
test_labels_path = None

# Try different possible locations
possible_paths = [
    ('/kaggle/input/mnist-dataset/train-images-idx3-ubyte/train-images-idx3-ubyte',
     '/kaggle/input/mnist-dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte',
     '/kaggle/input/mnist-dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte',
     '/kaggle/input/mnist-dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'),
    ('/kaggle/input/mnist-dataset/train-images.idx3-ubyte',
     '/kaggle/input/mnist-dataset/train-labels.idx1-ubyte',
     '/kaggle/input/mnist-dataset/t10k-images.idx3-ubyte',
     '/kaggle/input/mnist-dataset/t10k-labels.idx1-ubyte'),
]

# Try to find the correct paths
for tp in possible_paths:
    if all(os.path.isfile(p) for p in tp):
        train_images_path, train_labels_path, test_images_path, test_labels_path = tp
        break

# If still not found, search for any .ubyte files
if train_images_path is None:
    ubyte_files = glob.glob('/kaggle/input/mnist-dataset/**/*ubyte*', recursive=True)
    ubyte_files = [f for f in ubyte_files if os.path.isfile(f)]
    print("\nFound .ubyte files:")
    for f in ubyte_files:
        print(f"  {f}")
    
    # Try to match them
    for f in ubyte_files:
        fname = os.path.basename(f).lower()
        if 'train' in fname and 'images' in fname:
            train_images_path = f
        elif 'train' in fname and 'labels' in fname:
            train_labels_path = f
        elif 't10k' in fname and 'images' in fname:
            test_images_path = f
        elif 't10k' in fname and 'labels' in fname:
            test_labels_path = f

if train_images_path is None:
    raise FileNotFoundError("Could not locate MNIST data files. Please check the dataset structure.")

print(f"\nUsing files:")
print(f"  Train images: {train_images_path}")
print(f"  Train labels: {train_labels_path}")
print(f"  Test images:  {test_images_path}")
print(f"  Test labels:  {test_labels_path}")

# Load datasets from IDX format
print("\nLoading MNIST dataset from IDX files...")
train_images, train_labels = load_mnist_idx(train_images_path, train_labels_path)
test_images, test_labels = load_mnist_idx(test_images_path, test_labels_path)

print(f"Training samples: {len(train_images)}")
print(f"Test samples: {len(test_images)}")
print(f"Image shape: {train_images[0].shape}")

# Create datasets
train_dataset = MNISTDataset(train_images, train_labels, transform=transform_cnn)
test_dataset = MNISTDataset(test_images, test_labels, transform=transform_cnn)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Visualize some samples
print("\nVisualizing sample images...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    ax = axes[i//5, i%5]
    ax.imshow(train_images[i], cmap='gray')
    ax.set_title(f'Label: {train_labels[i]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nDataset loaded successfully!")

########################### PART 1.1: CNN Architecture ###########################

class CNN_MNIST(nn.Module):
    """Custom CNN for MNIST Classification"""
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # 7x7 -> 3x3
        x = self.dropout1(x)
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        
        return x

# ==================================================================================
# Training and Evaluation Functions
# ==================================================================================

def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    """Train the model"""
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies

def evaluate_model(model, test_loader, device):
    """Evaluate the model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1, all_preds, all_labels

# ==================================================================================
# Train CNN Model
# ==================================================================================

print("\n" + "="*80)
print("TRAINING CNN MODEL")
print("="*80)

cnn_model = CNN_MNIST().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001, weight_decay=1e-5)

start_time = time.time()
cnn_losses, cnn_accs = train_model(cnn_model, train_loader, criterion, optimizer, device, epochs=10)
cnn_train_time = time.time() - start_time

print(f"\nCNN Training Time: {cnn_train_time:.2f} seconds")

# Evaluate CNN
cnn_accuracy, cnn_f1, cnn_preds, cnn_labels = evaluate_model(cnn_model, test_loader, device)
print(f"CNN Test Accuracy: {cnn_accuracy*100:.2f}%")
print(f"CNN F1 Score: {cnn_f1:.4f}")

################### PART 1.2: Faster R-CNN (Adapted for MNIST) #########################

print("\n" + "="*80)
print("NOTE: Faster R-CNN for Classification")
print("="*80)
print("Faster R-CNN is designed for object detection, not classification.")
print("We'll create a CNN-based classifier with similar depth for comparison.")

class DeepCNN_MNIST(nn.Module):
    """Deeper CNN similar to detection backbone"""
    def __init__(self):
        super(DeepCNN_MNIST, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Train Deep CNN
print("\nTraining Deep CNN (R-CNN style architecture)...")
rcnn_model = DeepCNN_MNIST().to(device)
optimizer_rcnn = optim.Adam(rcnn_model.parameters(), lr=0.001, weight_decay=1e-5)

start_time = time.time()
rcnn_losses, rcnn_accs = train_model(rcnn_model, train_loader, criterion, optimizer_rcnn, device, epochs=10)
rcnn_train_time = time.time() - start_time

print(f"\nDeep CNN Training Time: {rcnn_train_time:.2f} seconds")

# Evaluate R-CNN style model
rcnn_accuracy, rcnn_f1, rcnn_preds, rcnn_labels = evaluate_model(rcnn_model, test_loader, device)
print(f"Deep CNN Test Accuracy: {rcnn_accuracy*100:.2f}%")
print(f"Deep CNN F1 Score: {rcnn_f1:.4f}")

########################### PART 1.4: Pretrained Models (VGG16 and AlexNet) #################################

print("\n" + "="*80)
print("FINE-TUNING PRETRAINED MODELS")
print("="*80)

# Reload data with pretrained transforms
train_dataset_pretrained = MNISTDataset(train_images, train_labels, transform=transform_pretrained)
test_dataset_pretrained = MNISTDataset(test_images, test_labels, transform=transform_pretrained)

train_loader_pretrained = DataLoader(train_dataset_pretrained, batch_size=32, shuffle=True, num_workers=2)
test_loader_pretrained = DataLoader(test_dataset_pretrained, batch_size=32, shuffle=False, num_workers=2)

# VGG16
print("\nFine-tuning VGG16...")
vgg_model = vgg16(pretrained=True)
# Freeze early layers
for param in vgg_model.features.parameters():
    param.requires_grad = False

# Modify classifier
vgg_model.classifier[6] = nn.Linear(4096, 10)
vgg_model = vgg_model.to(device)

optimizer_vgg = optim.Adam(vgg_model.classifier.parameters(), lr=0.0001)

start_time = time.time()
vgg_losses, vgg_accs = train_model(vgg_model, train_loader_pretrained, criterion, optimizer_vgg, device, epochs=5)
vgg_train_time = time.time() - start_time

vgg_accuracy, vgg_f1, _, _ = evaluate_model(vgg_model, test_loader_pretrained, device)
print(f"\nVGG16 Training Time: {vgg_train_time:.2f} seconds")
print(f"VGG16 Test Accuracy: {vgg_accuracy*100:.2f}%")
print(f"VGG16 F1 Score: {vgg_f1:.4f}")

# AlexNet
print("\n\nFine-tuning AlexNet...")
alex_model = alexnet(pretrained=True)
# Freeze early layers
for param in alex_model.features.parameters():
    param.requires_grad = False

alex_model.classifier[6] = nn.Linear(4096, 10)
alex_model = alex_model.to(device)

optimizer_alex = optim.Adam(alex_model.classifier.parameters(), lr=0.0001)

start_time = time.time()
alex_losses, alex_accs = train_model(alex_model, train_loader_pretrained, criterion, optimizer_alex, device, epochs=5)
alex_train_time = time.time() - start_time

alex_accuracy, alex_f1, _, _ = evaluate_model(alex_model, test_loader_pretrained, device)
print(f"\nAlexNet Training Time: {alex_train_time:.2f} seconds")
print(f"AlexNet Test Accuracy: {alex_accuracy*100:.2f}%")
print(f"AlexNet F1 Score: {alex_f1:.4f}")

######################## PART 2: Vision Transformer (ViT) ############################
print("\n" + "="*80)
print("VISION TRANSFORMER FROM SCRATCH")
print("="*80)

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer for MNIST"""
    def __init__(self, img_size=28, patch_size=4, in_channels=1, num_classes=10,
                 embed_dim=64, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x

# Train Vision Transformer
print("\nTraining Vision Transformer...")
vit_model = VisionTransformer(
    img_size=28,
    patch_size=4,
    in_channels=1,
    num_classes=10,
    embed_dim=128,
    depth=6,
    num_heads=8,
    mlp_ratio=4.0,
    dropout=0.1
).to(device)

optimizer_vit = optim.AdamW(vit_model.parameters(), lr=0.001, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_vit, T_max=10)

start_time = time.time()
vit_losses, vit_accs = train_model(vit_model, train_loader, criterion, optimizer_vit, device, epochs=10)
vit_train_time = time.time() - start_time

vit_accuracy, vit_f1, _, _ = evaluate_model(vit_model, test_loader, device)
print(f"\nViT Training Time: {vit_train_time:.2f} seconds")
print(f"ViT Test Accuracy: {vit_accuracy*100:.2f}%")
print(f"ViT F1 Score: {vit_f1:.4f}")


################# PART 3: Comparison and Visualization #########################


print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

results = pd.DataFrame({
    'Model': ['CNN', 'Deep CNN', 'VGG16', 'AlexNet', 'ViT'],
    'Accuracy (%)': [cnn_accuracy*100, rcnn_accuracy*100, vgg_accuracy*100, 
                     alex_accuracy*100, vit_accuracy*100],
    'F1 Score': [cnn_f1, rcnn_f1, vgg_f1, alex_f1, vit_f1],
    'Training Time (s)': [cnn_train_time, rcnn_train_time, vgg_train_time, 
                          alex_train_time, vit_train_time]
})

print("\n", results.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy comparison
axes[0].bar(results['Model'], results['Accuracy (%)'], color=['blue', 'green', 'red', 'orange', 'purple'])
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylim([95, 100])
axes[0].grid(axis='y', alpha=0.3)

# F1 Score comparison
axes[1].bar(results['Model'], results['F1 Score'], color=['blue', 'green', 'red', 'orange', 'purple'])
axes[1].set_ylabel('F1 Score')
axes[1].set_title('Model F1 Score Comparison')
axes[1].set_ylim([0.95, 1.0])
axes[1].grid(axis='y', alpha=0.3)

# Training time comparison
axes[2].bar(results['Model'], results['Training Time (s)'], color=['blue', 'green', 'red', 'orange', 'purple'])
axes[2].set_ylabel('Time (seconds)')
axes[2].set_title('Training Time Comparison')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
