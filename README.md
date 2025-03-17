# ISIC 2016 Skin Lesion Segmentation Using MobileNetV2

This repository contains implementation of a semantic segmentation model for skin lesion segmentation based on the ISIC 2016 dataset. The model uses MobileNetV2 as the encoder backbone with a custom decoder architecture.

## Objective

The main objective of this project is to train segmentation models using a pre-trained MobileNetV2 encoder with a custom decoder for skin lesion segmentation, exploring two different training strategies:

1. **Feature Extraction**: Training only the decoder while keeping the pre-trained encoder frozen
2. **Fine-tuning**: Training both the encoder and decoder with different learning rates

## [Dataset] (https://www.kaggle.com/datasets/mahmudulhasantasin/isic-2016-original-dataset)

The project uses the ISIC 2016 skin lesion dataset which contains:
- 900 training images with corresponding segmentation masks
- 379 test images with corresponding segmentation masks

All images are preprocessed and resized to 128x128 pixels.

## Project Structure

```
isic-segmentation-mobilenet/
├── data/                   # Dataset directory (not included in repo)
├── models/                 # Model architecture definitions
├── utils/                  # Utility functions for data loading and metrics
├── notebooks/              # Analysis notebooks
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── visualize.py            # Visualization script
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/mitesh-kr/ISIC-Vision--Image-Segmentation-with-MobileNet-Encoder.git
cd ISIC-Vision--Image-Segmentation-with-MobileNet-Encoder

```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the ISIC 2016 dataset and place it in the `data/` directory with the following structure:
```
data/
├── train/          # Training images
├── train_masks/    # Training masks
├── test/           # Test images
└── test_masks/     # Test masks
```

## Usage

### Training

To train the model with frozen encoder (Experiment 1):
```bash
python train.py --mode feature_extraction --epochs 50
```

To train the model with encoder fine-tuning (Experiment 2):
```bash
python train.py --mode finetuning --epochs 50
```

### Evaluation

To evaluate a trained model:
```bash
python evaluate.py --model_path path/to/saved/model.pth
```

### Visualization

To visualize predictions:
```bash
python visualize.py --model_path path/to/saved/model.pth
```

## Model Architecture

- **Encoder**: Pre-trained MobileNetV2 trained on ImageNet
- **Decoder**: Custom decoder with 9 layers (half the number in the encoder)
- **Skip Connections**: Total of 9 skip connections between encoder and decoder to handle vanishing gradients
- **Output Size**: Upsamples from 1280×4×4 to 1×128×128
- **Regularization**: Dropout layers to avoid overfitting

## Experiments and Results

Two main experiments were conducted:

### Experiment 1: Feature Extraction
- Encoder: Pre-trained MobileNetV2 (frozen)
- Decoder: Custom U-Net style decoder
- Training: Only decoder parameters (encoder weights frozen)
- Encoder learning rate: 0
- Decoder learning rate: 0.01
- Loss function: Binary Cross Entropy
- Batch size: 45
- Number of epochs: 50

### Experiment 2: Fine-tuning
- Encoder: Pre-trained MobileNetV2 (fine-tuned)
- Decoder: Custom U-Net style decoder
- Training: Both encoder and decoder parameters
- Encoder learning rate: 0.0001
- Decoder learning rate: 0.01
- Loss function: Binary Cross Entropy
- Batch size: 45
- Number of epochs: 50

### Performance Metrics
Both models were evaluated using:
- Training Loss
- Validation Loss
- Intersection over Union (IoU)
- Dice Coefficient

### Comparative Analysis

| Metric | Model 1 (Feature Extraction) | Model 2 (Fine-tuning) |
|--------|------------------------------|------------------------|
| Final Training Loss | 0.1848 | 0.1658 |
| Final Validation Loss | 0.1630 | 0.1248 |
| Final IoU | 0.6454 | 0.7239 |
| Final Dice Score | 0.7641 | 0.8295 |
| Trainable Parameters | 1,594,314 | 3,818,186 |
| Non-trainable Parameters | 2,223,872 | 0 |

## Key Findings

1. **Model Performance**: The fine-tuning approach (Model 2) showed significantly better performance compared to feature extraction (Model 1), with higher IoU and Dice scores.

2. **Loss Comparison**: Both training and validation losses for Model 2 (fine-tuning) were lower than Model 1 (feature extraction), indicating better model fit and generalization.

3. **Metric Improvements**: Model 2 achieved approximately 7.85% higher IoU and 6.54% higher Dice score compared to Model 1.

4. **Visual Results**: Qualitative analysis of the segmentation results showed that Model 2 produced masks that more closely match the ground truth, especially in difficult cases with irregular boundaries or lower contrast.

5. **Thresholding Effect**: Applying a threshold of 0.5 to the predicted masks helped in obtaining cleaner binary segmentations for both models.

## Conclusion

Fine-tuning the pre-trained MobileNetV2 encoder (Model 2) yields superior segmentation performance compared to using a frozen encoder (Model 1). This demonstrates the importance of allowing the encoder to adapt to the specific features of the skin lesion dataset, even when starting from a pre-trained state.

The results highlight the effectiveness of the U-Net-style architecture with skip connections in preserving spatial information throughout the network, which is crucial for accurate segmentation boundaries.

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses the ISIC 2016 skin lesion dataset
- MobileNetV2 implementation from torchvision
