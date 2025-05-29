# YOLO Dataset Transformation for Primate Call Detection

This document describes the utilities for transforming primate call datasets into YOLO-format tensors suitable for object detection in spectrograms.

## Overview

The YOLO (You Only Look Once) format represents object detection as a regression problem by dividing the input into an S × S grid. For each grid cell, the model predicts:
- B bounding boxes with confidence scores
- C class probabilities

The output tensor has dimensions S × S × (B × 5 + C), where:
- S = Grid size (default: 7)
- B = Number of bounding boxes per cell (default: 2)  
- C = Number of classes
- Each bounding box has 5 values: [x_center, y_center, width, height, confidence]

## Dataset Format

The input CSV should contain the following columns:
- `begin_time`: Start time of the call (seconds)
- `end_time`: End time of the call (seconds)
- `low_freq`: Lower frequency bound (Hz)
- `high_freq`: Upper frequency bound (Hz)
- `species`: Species identifier (e.g., 'lw', 'aa', 'ac')
- `call_type`: Call type identifier (e.g., 'cs', 'cc', 'tr')
- `recording_file`: Name of the audio recording file
- `inband_power`: Power measurement (optional)

## Classes

Classes are automatically created from unique combinations of `species` and `call_type`. For example:
- 'lw' + 'cs' → 'lw_cs'
- 'aa' + 'tr' → 'aa_tr'

## Usage Examples

### Basic Usage

```python
from banana_net.utils.data import create_yolo_dataset, analyze_dataset_statistics

# Analyze dataset first
stats = analyze_dataset_statistics("data/processed/call_dataset_new.csv")
print(f"Found {stats['unique_classes']} classes")

# Create YOLO tensors
yolo_tensors, class_info = create_yolo_dataset(
    csv_path="data/processed/call_dataset_new.csv",
    grid_size=7,
    num_boxes=2,
    spectrogram_duration=10.0,
    freq_range=(0, 22050)
)

# Access tensors by recording file
for recording_file, tensor in yolo_tensors.items():
    print(f"{recording_file}: {tensor.shape}")
```

### Advanced Usage with Custom Parameters

```python
from banana_net.utils.data import YOLODatasetTransformer
import pandas as pd

# Create transformer with custom parameters
transformer = YOLODatasetTransformer(
    grid_size=14,  # Higher resolution grid
    num_boxes=3,   # More boxes per cell
    spectrogram_duration=15.0,  # Longer spectrograms
    freq_range=(0, 44100)  # Higher frequency range
)

# Load and process dataset
df = pd.read_csv("data/processed/call_dataset_new.csv")
transformer.build_class_mapping(df)

# Process specific recording
annotations = df.to_dict('records')
tensor = transformer.create_yolo_tensor(annotations, "specific_recording.wav")

# Save class mapping for later use
transformer.save_class_mapping("class_mapping.json")
```

### Loading Saved Class Mapping

```python
# Load previously saved class mapping
transformer = YOLODatasetTransformer()
transformer.load_class_mapping("class_mapping.json")

# Use with new data (must have same classes)
new_tensor = transformer.create_yolo_tensor(new_annotations, "new_recording.wav")
```

## Coordinate System

### Input Coordinates
- Time: Absolute seconds from recording start
- Frequency: Absolute Hz values

### Normalized Coordinates
- Time: [0, 1] relative to `spectrogram_duration`
- Frequency: [0, 1] relative to max frequency in `freq_range`

### Grid Cell Coordinates
- Relative to individual grid cells
- Bounding box centers and dimensions scaled by grid size

## Tensor Structure

For default parameters (S=7, B=2), each tensor has shape (7, 7, 10+C):

```
Channels 0-4:   First bounding box [x, y, w, h, confidence]
Channels 5-9:   Second bounding box [x, y, w, h, confidence]
Channels 10+:   Class probabilities (one-hot encoded)
```

## Validation and Quality Checks

The implementation includes several validation features:

1. **Coordinate Clamping**: Ensures all coordinates stay in [0, 1] range
2. **Grid Bounds Checking**: Prevents out-of-bounds grid cell access
3. **Class Validation**: Warns about unknown classes during processing
4. **Multiple Objects**: Handles multiple objects in the same grid cell

## Performance Considerations

- **Memory Usage**: Tensors grow quadratically with grid size
- **Processing Speed**: Linear with number of annotations
- **Class Count**: Affects tensor depth and memory usage

## Integration with Training

The output tensors are ready for use with PyTorch-based YOLO training:

```python
import torch
import torch.nn as nn

# Example loss calculation (simplified)
def yolo_loss(predictions, targets):
    # predictions: (batch_size, S, S, B*5 + C)
    # targets: (batch_size, S, S, B*5 + C)
    
    # Separate bounding boxes and class predictions
    pred_boxes = predictions[..., :B*5]
    pred_classes = predictions[..., B*5:]
    
    target_boxes = targets[..., :B*5]
    target_classes = targets[..., B*5:]
    
    # Calculate losses (coordinate, confidence, classification)
    # ... loss computation logic
    
    return total_loss
```

## Common Issues and Solutions

### 1. Empty Tensors
**Problem**: All confidence values are 0
**Solution**: Check if recording files in CSV match the queried recording name

### 2. Coordinate Errors
**Problem**: Objects appear in wrong locations
**Solution**: Verify `spectrogram_duration` and `freq_range` match your audio analysis

### 3. Class Mapping Errors
**Problem**: Unknown class warnings
**Solution**: Ensure all test data classes exist in training data

### 4. Memory Issues
**Problem**: Out of memory errors
**Solution**: Reduce `grid_size` or process recordings in batches

## Future Enhancements

Potential improvements to the current implementation:

1. **Multi-scale Detection**: Support for different grid sizes
2. **Anchor Boxes**: Pre-defined aspect ratios for better detection
3. **Data Augmentation**: Built-in augmentation for training
4. **Batch Processing**: Efficient processing of multiple recordings
5. **Visualization Tools**: Built-in plotting for debugging

## References

- Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection" (2016)
- Original YOLO paper implementation details
- PyTorch object detection tutorials
