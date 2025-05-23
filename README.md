# CenterNet Object Detection with PASCAL VOC 2007

This repository contains a PyTorch implementation of CenterNet object detection model trained on PASCAL VOC 2007 dataset with MPS (Metal Performance Shaders) support for Mac devices.

## Features

- CenterNet architecture implementation
- PASCAL VOC 2007 dataset support with automatic download
- PyTorch MPS support for Mac M1/M2 devices
- ResNet18/ResNet50 backbone options
- Complete training and evaluation pipeline
- Visualization of detection results

## Requirements

```bash
pip install torch torchvision
pip install numpy matplotlib pillow opencv-python
```

## Dataset

The code automatically downloads PASCAL VOC 2007 dataset when run for the first time. The dataset contains:
- 20 object classes
- ~5,000 training images
- ~5,000 test images

## Architecture

### CenterNet Model
- **Backbone**: ResNet18 (default) or ResNet50
- **Head Networks**: 
  - Heatmap head for object centers
  - Width/Height regression head
  - Offset regression head
- **Loss Functions**:
  - Focal loss for heatmap
  - L1 loss for width/height and offset regression

## Usage

### Training

```python
python train_centernet_voc2007.py
```

The training script will:
1. Download VOC 2007 dataset automatically
2. Train CenterNet model for 30 epochs
3. Save checkpoints every 5 epochs
4. Save the best model based on validation accuracy

### Model Configuration

Key parameters in the code:
- `input_size`: 512x512 (default)
- `batch_size`: 8
- `learning_rate`: 1e-4
- `num_epochs`: 30
- `backbone`: 'resnet18' or 'resnet50'

### Output

- Checkpoints saved in `./checkpoints/` directory
- Detection results saved in `./results/` directory
- Training logs printed to console

## Model Performance

The model uses a more lenient evaluation criteria:
- IoU threshold: 0.3
- Score threshold: 0.01

This allows for better detection of small objects which are common in VOC dataset.

## Code Structure

```
train_centernet_voc2007.py
├── Dataset Classes
│   └── VOCDetectionDataset: Wrapper for VOC dataset with preprocessing
├── Model Architecture
│   ├── CenterNet: Main model class
│   └── CenterNetHead: Detection head networks
├── Loss Functions
│   ├── FocalLoss: For heatmap classification
│   └── RegL1Loss: For regression tasks
├── Training Functions
│   ├── train_one_epoch: Single epoch training
│   └── evaluate: Model evaluation
├── Utility Functions
│   ├── gaussian2D: Generate gaussian kernels
│   ├── decode: Post-processing for detections
│   └── visualize_prediction: Result visualization
└── Main Functions
    ├── main: Training pipeline
    └── test_model: Testing and inference
```

## VOC Classes

The model detects 20 object classes:
- aeroplane, bicycle, bird, boat, bottle
- bus, car, cat, chair, cow
- diningtable, dog, horse, motorbike, person
- pottedplant, sheep, sofa, train, tvmonitor

## MPS Support

The code automatically detects and uses MPS (Metal Performance Shaders) on compatible Mac devices:
```python
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
```

## Visualization

After training, the model automatically runs inference on a test image and saves the visualization with:
- Bounding boxes in red
- Class names and confidence scores
- Results saved in `./results/` directory

## Tips for Better Performance

1. **For faster training**: Use ResNet18 backbone (default)
2. **For better accuracy**: Use ResNet50 backbone and increase epochs
3. **Memory issues**: Reduce batch_size
4. **Better small object detection**: Adjust gaussian_radius parameters

## Troubleshooting

- **CUDA out of memory**: Reduce batch_size
- **MPS errors on Mac**: Update PyTorch to latest version
- **Dataset download fails**: Check internet connection and disk space

## References

- Original CenterNet paper: [Objects as Points](https://arxiv.org/abs/1904.07850)
- PASCAL VOC dataset: [Official Website](http://host.robots.ox.ac.uk/pascal/VOC/)

## License

This project is open source and available under the MIT License.