# Lab 2: Deep Learning with PyTorch on MNIST Dataset

## Objective
The main purpose of this lab is to get familiar with the PyTorch library, build neural architectures such as CNN, R-CNN (adapted), FCNN, ViT, etc., for computer vision tasks, specifically classification on the MNIST dataset.

## Tools Used
- Google Colab or Kaggle for development and training.
- PyTorch for model implementation.
- GitHub for version control and reporting.

## Dataset
- **MNIST Dataset**: Sourced from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
- Contains 60,000 training images and 10,000 test images of handwritten digits (0-9), each 28x28 pixels in grayscale.

## Work Done

### Part 1: CNN Classifier

#### 1. Establishing a CNN Architecture
- Built a custom CNN using PyTorch for MNIST classification.
- Architecture includes:
  - Convolutional layers: 3 layers (32, 64, 128 filters) with 3x3 kernels, padding=1.
  - Activation: ReLU.
  - Pooling: MaxPool2d (2x2, stride=2).
  - Batch Normalization after each conv layer.
  - Dropout: 0.25 after pooling, 0.5 in FC layers.
  - Fully Connected layers: 128*3*3 → 256 → 128 → 10.
- Hyper-parameters:
  - Optimizer: Adam (lr=0.001, weight_decay=1e-5).
  - Loss: CrossEntropyLoss.
  - Batch size: 64.
  - Epochs: 10.
  - Trained on GPU (CUDA if available).
- Data transformations: Normalize(mean=0.1307, std=0.3081).

#### 2. Faster R-CNN
- Faster R-CNN is primarily for object detection, so adapted it to a deeper CNN classifier for fair comparison (referred to as "Deep CNN" in code).
- Architecture: Sequential conv blocks with more depth (64→64→128→128→256→256 filters), followed by FC layers (256*3*3 → 512 → 10).
- Similar hyper-parameters as CNN, trained for 10 epochs.

#### 3. Comparison of CNN and Deep CNN (Faster R-CNN Adapted)
- Metrics: Accuracy, F1 Score, Loss (during training), Training Time.
- Evaluated on test set using scikit-learn metrics.

#### 4. Fine-Tuning Pretrained Models (VGG16 and AlexNet)
- Used transfer learning on pretrained models from torchvision.
- Data transformations: Resize to 224x224, Grayscale to 3 channels, Normalize(ImageNet stats).
- Froze feature extractor layers, fine-tuned classifiers.
- Modified final FC layer to output 10 classes.
- Optimizer: Adam (lr=0.0001) on classifier params.
- Epochs: 5 (to reduce training time due to larger input size).
- Compared results to CNN and Deep CNN.
- Conclusion: Pretrained models perform well with transfer learning, but require more training time due to larger architectures. VGG16 and AlexNet show strong accuracy on MNIST despite being trained on natural images, highlighting the effectiveness of feature reuse.

### Part 2: Vision Transformer (ViT)
- Followed tutorial: [Vision Transformers from Scratch (PyTorch): A Step-by-Step Guide](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c).
- Built ViT from scratch for MNIST classification.
- Key components:
  - Patch Embedding: Image split into 4x4 patches (49 patches for 28x28), embedded to dim=128.
  - CLS Token and Positional Embeddings.
  - Transformer Blocks: 6 layers, 8 heads, MLP ratio=4.0, dropout=0.1.
  - Attention: Multi-Head Self-Attention.
  - Final: LayerNorm → Linear to 10 classes.
- Hyper-parameters:
  - Optimizer: AdamW (lr=0.001, weight_decay=0.05).
  - Scheduler: CosineAnnealingLR.
  - Epochs: 10.
- Interpreted results: ViT achieves competitive performance but may require more tuning for small datasets like MNIST compared to CNNs.
- Compared to Part 1: ViT shows slightly lower accuracy but demonstrates attention-based learning's potential.

## Results
All models were trained and evaluated on the MNIST test set. Below is a summary table of key metrics.

| Model     | Accuracy (%) | F1 Score | Training Time (s) |
|-----------|--------------|----------|-------------------|
| CNN       | 99.38       | 0.9938  | 88.29            |
| Deep CNN  | 99.40       | 0.9940  | 101.16           |
| VGG16     | 99.08       | 0.9908  | 996.34           |
| AlexNet   | 99.26       | 0.9926  | 333.10           |
| ViT       | 97.97       | 0.9797  | 189.49           |

<img width="1280" height="482" alt="image" src="https://github.com/user-attachments/assets/d9eed661-6b27-4b02-9b52-1b31de4b3854" />


- **Accuracy and F1 Score**: All models perform exceptionally well (>97%), with Deep CNN slightly edging out others. ViT is competitive but lags slightly, possibly due to the small image size and dataset simplicity.
- **Training Time**: Custom CNN and Deep CNN are fastest. Pretrained models (VGG16, AlexNet) take longer due to larger input sizes and architectures. ViT balances speed and performance.
- Training curves (losses and accuracies) were monitored, showing steady convergence for all models.

## Conclusions
1. **CNN vs. Deep CNN**: Deeper architectures provide marginal improvements in feature extraction but increase training time slightly.
2. **Pretrained Models**: Transfer learning from VGG16 and AlexNet yields high accuracy with minimal fine-tuning, proving effective even for grayscale digits.
3. **Vision Transformer**: ViT performs well, showcasing the power of attention mechanisms, but CNNs are more efficient for simple tasks like MNIST.
4. **Trade-offs**: For production, choose based on accuracy vs. efficiency. CNNs are lightweight and fast; ViT offers scalability for complex vision tasks.
5. Overall, PyTorch simplifies building and training diverse architectures, and MNIST serves as an excellent benchmark for comparing models.

## Synthesis: What I Learned
During this lab, I gained hands-on experience with PyTorch for building and training neural networks from scratch, including CNNs, adapted detection models, pretrained fine-tuning, and Transformers. I learned about hyper-parameter tuning, data preprocessing, GPU acceleration, and performance metrics. Key takeaways include understanding trade-offs between model complexity and efficiency, the benefits of transfer learning, and how attention mechanisms in ViT differ from convolutional approaches. This deepened my knowledge of computer vision and prepared me for more advanced deep learning projects.

## Repository Structure
- `all.py`: Complete code for data loading, model implementations, training, evaluation, and visualizations.
- `model_comparison.png`: Generated comparison charts.
- `sample_images.png`: Sample MNIST visualizations (generated in code).
- `README.md`: This report.

For full code execution, run in a Kaggle notebook with the MNIST dataset attached.
