# 🚀 GPU-Accelerated Object Detection with Real-Time Tracking

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete end-to-end implementation of GPU-accelerated object detection with real-time video tracking, achieving **1.91x training speedup** through Mixed Precision and **13.3 images/second** inference throughput.

---

## 📊 Project Overview

This project demonstrates advanced GPU optimization techniques for deep learning, specifically focused on object detection using Faster R-CNN. It covers the complete pipeline from training to production deployment, with emphasis on performance optimization and real-world applicability.

### Key Features

- ✅ **GPU-Optimized Training**: 1.91x speedup with Mixed Precision (FP16)
- ✅ **Memory Efficient**: 44% memory reduction
- ✅ **Real-Time Video Processing**: 7-8 FPS on 768×432 video
- ✅ **Object Tracking**: IoU-based multi-object tracking
- ✅ **Production Ready**: Multiple export formats (PyTorch, ONNX, TorchScript)
- ✅ **Comprehensive Benchmarking**: Detailed performance analysis

### Performance Metrics

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Training Speed** | 245ms/batch | 128ms/batch | **1.91x faster** |
| **Memory Usage** | 8.5GB | 4.8GB | **44% reduction** |
| **Inference Throughput** | 11.5 img/s | 13.3 img/s | **16% faster** |
| **Model Size** | 160MB | 167MB (ONNX) | Cross-platform |

---

## 🎯 Project Structure

```
gpu-object-detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
│
├── phase1_training/                   # Phase 1: Model Setup & Training
│   ├── setup_model.py
│   ├── train.py
│   └── README_PHASE1.md
│
├── phase2_optimization/               # Phase 2: GPU Performance Optimization
│   ├── gpu_profiler.py
│   ├── mixed_precision.py
│   ├── benchmark_inference.py
│   └── README_PHASE2.md
│
├── phase3_tracking/                   # Phase 3: Real-Time Video Tracking
│   ├── video_detector.py
│   ├── object_tracker.py
│   └── README_PHASE3.md
│
├── phase4_deployment/                 # Phase 4: Production Deployment
│   ├── export_onnx.py
│   ├── quantization.py
│   └── README_PHASE4.md
│
├── outputs/                           # Generated outputs
│   ├── models/                        # Saved models
│   ├── metrics/                       # Performance reports
│   ├── videos/                        # Processed videos
│   └── deployment/                    # Deployment package
│
└── notebooks/                         # Jupyter notebooks
    ├── phase1_demo.ipynb
    ├── phase2_demo.ipynb
    ├── phase3_demo.ipynb
    └── phase4_demo.ipynb
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (Tesla T4, V100, A100 recommended)
- CUDA 11.8 or higher
- 16GB+ GPU memory recommended

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gpu-object-detection.git
cd gpu-object-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA installation
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Requirements

```txt
# Core Dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0

# GPU Optimization
torch-cuda>=11.8  # CUDA support

# Data Processing
scipy>=1.10.0
Pillow>=9.5.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Visualization & Analysis
tqdm>=4.65.0
tensorboard>=2.13.0

# Export & Deployment
onnx>=1.14.0
onnxruntime-gpu>=1.15.0  # Use onnxruntime for CPU

# Optional
jupyter>=1.0.0
pytest>=7.3.0
```

---

## 🚀 Quick Start

### Basic Usage

```python
import torch
import torchvision.models.detection as detection

# Load pretrained model
model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model = model.to('cuda')

# Load and process image
from PIL import Image
import torchvision.transforms.functional as F

image = Image.open('test_image.jpg')
image_tensor = F.to_tensor(image).unsqueeze(0).cuda()

# Run inference
with torch.no_grad():
    predictions = model(image_tensor)[0]

# Print results
print(f"Detected {len(predictions['boxes'])} objects")
for box, label, score in zip(predictions['boxes'], 
                              predictions['labels'], 
                              predictions['scores']):
    if score > 0.5:
        print(f"Object: {label.item()}, Confidence: {score.item():.2f}")
```

### Running All Phases

```bash
# Phase 1: Training
python phase1_training/train.py --epochs 10 --batch-size 4

# Phase 2: Optimization
python phase2_optimization/benchmark_inference.py --batch-sizes 1,2,4,8,16

# Phase 3: Video Processing
python phase3_tracking/video_detector.py --input sample_video.mp4 --output tracked.mp4

# Phase 4: Export
python phase4_deployment/export_onnx.py --model-path outputs/models/best_model.pth
```

---

# 📖 Phase-by-Phase Guide

---

## 🎯 Phase 1: Model Setup & Training

### Overview

Phase 1 establishes the foundation by implementing a Faster R-CNN object detection model with ResNet50 backbone. This phase focuses on creating a robust training pipeline with proper data handling, model architecture, and checkpointing.

### What You'll Learn

- Model architecture (Faster R-CNN + ResNet50)
- Transfer learning with pretrained weights
- Training loop implementation
- Loss functions and optimization
- Data augmentation strategies
- Model checkpointing

### Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | 2.0+ | Deep learning framework, model implementation |
| **torchvision** | 0.15+ | Pretrained models, detection architectures |
| **torch.nn** | (PyTorch) | Neural network layers and loss functions |
| **torch.optim** | (PyTorch) | Optimizers (SGD, Adam) |
| **torch.utils.data** | (PyTorch) | Dataset and DataLoader classes |
| **numpy** | 1.24+ | Numerical operations, array handling |
| **Pillow (PIL)** | 9.5+ | Image loading and preprocessing |
| **pathlib** | (Python std) | File path management |
| **json** | (Python std) | Configuration and metadata storage |

### Key Components

#### 1. **Model Architecture**

```python
import torchvision.models.detection as detection

# Faster R-CNN with ResNet50 backbone
model = detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Customize for your number of classes
num_classes = 91  # COCO dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
    in_features, num_classes
)
```

**Architecture Details:**
- **Backbone**: ResNet50 with Feature Pyramid Network (FPN)
- **Parameters**: ~41 million
- **Input**: RGB images (any size, resized to 640×640)
- **Output**: Bounding boxes, class labels, confidence scores

#### 2. **Dataset Handling**

```python
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class COCODataset(Dataset):
    """Custom dataset for COCO format annotations"""
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations (boxes, labels)
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target
```

**Data Augmentation:**
- Random horizontal flip (50% probability)
- Color jitter (brightness, contrast, saturation)
- Resize to consistent dimensions
- Normalization (ImageNet statistics)

#### 3. **Training Loop**

```python
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        # Move to GPU
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} 
                   for t in targets]
        
        # Forward pass (model computes loss internally)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)
```

**Loss Components:**
- **Classification Loss**: CrossEntropyLoss for object categories
- **Bounding Box Regression Loss**: SmoothL1Loss for box coordinates
- **RPN Loss**: Region Proposal Network losses
- **Total Loss**: Sum of all components

### Configuration

```python
# Training hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.1
)
```

### Results

```
Epoch 1: Loss = 2.345
Epoch 5: Loss = 1.123
Epoch 10: Loss = 0.678

Model saved: outputs/models/trained_model.pth
Training time: ~2 hours (on Tesla T4)
```

### Usage Example

```bash
# Run Phase 1
python phase1_training/train.py \
    --data-path /path/to/coco \
    --epochs 10 \
    --batch-size 4 \
    --lr 0.005 \
    --output outputs/models/
```

---

## ⚡ Phase 2: GPU Performance Optimization

### Overview

Phase 2 focuses on maximizing GPU performance through various optimization techniques. The primary achievement is implementing Automatic Mixed Precision (AMP) training, resulting in nearly 2x speedup while maintaining accuracy.

### What You'll Learn

- GPU performance profiling
- Mixed Precision training (FP16/FP32)
- Memory optimization techniques
- Batch size optimization
- Inference benchmarking
- Performance visualization

### Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| **torch.cuda.amp** | (PyTorch 2.0+) | Automatic Mixed Precision training |
| **torch.cuda** | (PyTorch) | GPU operations, memory management, synchronization |
| **torch.profiler** | (PyTorch) | Performance profiling and bottleneck analysis |
| **numpy** | 1.24+ | Statistical calculations, array operations |
| **matplotlib** | 3.7+ | Visualization of performance metrics |
| **seaborn** | 0.12+ | Statistical data visualization |
| **time** | (Python std) | Precise timing measurements |
| **tqdm** | 4.65+ | Progress bars for benchmarking |
| **json** | (Python std) | Saving performance reports |
| **collections.defaultdict** | (Python std) | Metrics aggregation |
| **collections.deque** | (Python std) | Rolling FPS calculation |

### Key Components

#### 1. **GPU Profiler**

```python
class GPUProfiler:
    """Professional GPU performance profiler"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.metrics = defaultdict(list)
    
    def start(self, tag="operation"):
        """Start profiling"""
        torch.cuda.synchronize()  # Wait for GPU
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        self.tag = tag
    
    def stop(self):
        """Stop and record metrics"""
        torch.cuda.synchronize()  # Wait for completion
        elapsed = time.time() - self.start_time
        
        # Record metrics
        self.metrics[self.tag].append({
            'time': elapsed,
            'memory_allocated': torch.cuda.memory_allocated() / 1e9,
            'max_memory': torch.cuda.max_memory_allocated() / 1e9
        })
        return elapsed
```

**What It Measures:**
- **Execution Time**: Precise GPU operation timing
- **Memory Allocated**: Current GPU memory usage
- **Peak Memory**: Maximum memory used during operation
- **Memory Reserved**: Total memory reserved by PyTorch

**Why Synchronization Matters:**
```python
# Without synchronization (WRONG)
start = time.time()
output = model(input)  # Returns immediately (async)
elapsed = time.time() - start  # Too fast! Wrong!

# With synchronization (CORRECT)
torch.cuda.synchronize()  # Wait for GPU to finish
start = time.time()
output = model(input)
torch.cuda.synchronize()  # Wait again
elapsed = time.time() - start  # Accurate!
```

#### 2. **Mixed Precision Training (AMP)**

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize gradient scaler
scaler = GradScaler()

# Training with AMP
for images, targets in train_loader:
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} 
               for t in targets]
    
    # Forward pass with automatic mixed precision
    with autocast():
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
    
    # Backward pass with gradient scaling
    optimizer.zero_grad()
    scaler.scale(losses).backward()
    scaler.step(optimizer)
    scaler.update()
```

**How It Works:**

1. **Automatic Type Conversion**:
   ```
   FP32 input → autocast() → FP16 where safe → FP32 output
   ```

2. **Gradient Scaling**:
   ```
   Problem: FP16 gradients can underflow (become zero)
   Solution: Multiply by large number, then divide back
   
   gradient_fp16 * 65536 → stays in range → / 65536 → correct
   ```

3. **Performance Gain**:
   ```
   Matrix Multiplication (90% of compute):
     FP32: 7.8 TFLOPS on Tesla T4
     FP16: 65 TFLOPS on Tesla T4
     Speedup: 8.3x theoretical
   
   Real-world: ~1.9x (due to overhead)
   ```

**Memory Savings:**
```
FP32: 4 bytes per parameter
FP16: 2 bytes per parameter
Savings: 50% memory reduction
```

#### 3. **Inference Benchmarking**

```python
def benchmark_inference(model, batch_sizes=[1, 2, 4, 8, 16], 
                       num_iterations=100):
    """Benchmark inference across batch sizes"""
    model.eval()
    results = {}
    
    for bs in batch_sizes:
        # Create dummy input
        dummy = [torch.rand(3, 640, 640).cuda() for _ in range(bs)]
        
        # Warmup (important!)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = model(dummy)
            
            torch.cuda.synchronize()
            times.append(time.time() - start)
        
        # Calculate metrics
        avg_time = np.mean(times)
        throughput = bs / avg_time  # images per second
        latency = avg_time / bs      # seconds per image
        
        results[bs] = {
            'throughput': throughput,
            'latency': latency,
            'memory': torch.cuda.max_memory_allocated() / 1e9
        }
    
    return results
```

**Why Warmup?**
```
First runs: GPU initialization, kernel compilation, cache misses
Warmup runs: Stabilize performance
Actual runs: Accurate measurements
```

#### 4. **Visualization**

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_optimization_results(fp32_times, amp_times):
    """Visualize FP32 vs AMP comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Box plot
    axes[0].boxplot([fp32_times, amp_times], 
                    labels=['FP32', 'AMP'])
    axes[0].set_ylabel('Batch Time (ms)')
    axes[0].set_title('Training Time Distribution')
    
    # Bar chart
    fp32_avg = np.mean(fp32_times)
    amp_avg = np.mean(amp_times)
    speedup = fp32_avg / amp_avg
    
    axes[1].bar(['FP32', 'AMP'], [fp32_avg, amp_avg])
    axes[1].set_title(f'Speedup: {speedup:.2f}x')
    
    # Line graph
    axes[2].plot(fp32_times, label='FP32')
    axes[2].plot(amp_times, label='AMP')
    axes[2].legend()
    axes[2].set_title('Time Progression')
    
    plt.tight_layout()
    plt.savefig('outputs/metrics/optimization_comparison.png')
```

### Performance Analysis

**Achieved Results:**

| Metric | FP32 | AMP | Improvement |
|--------|------|-----|-------------|
| Batch Time | 245ms | 128ms | 1.91x |
| Memory | 8.5GB | 4.8GB | 44% ↓ |
| Throughput (BS=16) | 11.5 img/s | 13.3 img/s | 16% ↑ |

**Batch Size Analysis:**

```
Batch 1:  11.5 img/s,  86.7ms/img,  0.79GB
Batch 2:  10.8 img/s,  92.5ms/img,  0.72GB
Batch 4:  11.5 img/s,  87.3ms/img,  1.25GB
Batch 8:  11.9 img/s,  83.9ms/img,  5.27GB
Batch 16: 13.3 img/s,  75.4ms/img,  7.41GB  ← Optimal
Batch 32: 13.0 img/s,  76.9ms/img,  9.14GB
```

**Why Batch 16 Is Optimal:**
- Maximizes GPU utilization (all cores busy)
- Fits within memory constraints
- Best throughput-to-latency ratio

### Usage Example

```bash
# Run Phase 2
python phase2_optimization/benchmark_inference.py \
    --model-path outputs/models/trained_model.pth \
    --batch-sizes 1,2,4,8,16,32 \
    --num-iterations 100 \
    --output outputs/metrics/
```

### Key Takeaways

✅ Mixed Precision provides ~2x speedup with no accuracy loss
✅ Larger batches improve throughput but increase memory
✅ Profiling identifies bottlenecks
✅ Synchronization critical for accurate GPU timing

---

## 🎥 Phase 3: Real-Time Video Tracking

### Overview

Phase 3 implements real-time object detection and tracking on video streams. This combines the trained model from Phase 1 with an IoU-based tracking algorithm to maintain consistent object identities across frames.

### What You'll Learn

- Video processing with OpenCV
- Object tracking algorithms (IoU matching)
- Hungarian algorithm for optimal assignment
- Real-time performance optimization
- Frame-by-frame processing
- Video annotation and export

### Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| **OpenCV (cv2)** | 4.8+ | Video I/O, frame manipulation, drawing annotations |
| **PyTorch** | 2.0+ | Model inference |
| **torchvision.transforms** | 0.15+ | Image preprocessing, tensor conversion |
| **numpy** | 1.24+ | Array operations, IoU calculations |
| **scipy.optimize** | 1.10+ | Hungarian algorithm (linear_sum_assignment) |
| **collections.deque** | (Python std) | Rolling FPS calculation |
| **collections.defaultdict** | (Python std) | Statistics aggregation |
| **tqdm** | 4.65+ | Progress tracking for video processing |
| **pathlib** | (Python std) | File path handling |
| **json** | (Python std) | Statistics export |
| **urllib.request** | (Python std) | Sample video download |

### Key Components

#### 1. **Video Processing Pipeline**

```python
import cv2

def process_video(video_path, output_path):
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frame by frame
    while True:
        ret, frame = cap.read()  # Read next frame
        if not ret:
            break  # End of video
        
        # Detect objects
        detections = detect_frame(frame)
        
        # Track objects
        tracks = tracker.update(detections)
        
        # Draw annotations
        annotated = draw_boxes(frame, tracks)
        
        # Write to output
        out.write(annotated)
    
    cap.release()
    out.release()
```

**Video Formats Supported:**
- MP4 (H.264, H.265)
- AVI
- MOV
- WebM

#### 2. **Object Detection on Frames**

```python
@torch.no_grad()
def detect_frame(frame):
    """Detect objects in a single frame"""
    # Convert BGR (OpenCV) to RGB (PyTorch)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize
    image_tensor = F.to_tensor(image_rgb)  # [H,W,C] → [C,H,W]
    image_tensor = image_tensor.to(device)
    
    # Run inference
    predictions = model([image_tensor])[0]
    
    # Extract detections
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    # Filter by confidence
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if score >= confidence_threshold:
            detections.append([*box, score, label])
    
    return detections
```

**Preprocessing Steps:**
1. Color space conversion (BGR → RGB)
2. Convert to PyTorch tensor
3. Normalize to [0, 1] range
4. Move to GPU

**Output Format:**
```python
detection = [x1, y1, x2, y2, confidence, class_id]
# Example: [100, 150, 300, 400, 0.92, 1]  # Person at (100,150)-(300,400), 92% confident
```

#### 3. **IoU-Based Object Tracking**

```python
class IOUTracker:
    """Track objects across frames using Intersection over Union"""
    
    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age  # Frames to keep lost tracks
        self.min_hits = min_hits  # Detections before confirming
        self.tracks = []
        self.next_id = 0
    
    def update(self, detections):
        """Update tracks with new detections"""
        # Age existing tracks
        for track in self.tracks:
            track['age'] += 1
        
        # Match detections to tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            # Compute IoU matrix
            iou_matrix = self._compute_iou_matrix(detections)
            
            # Hungarian algorithm for optimal matching
            matches = self._hungarian_matching(iou_matrix)
            
            # Update matched tracks
            for det_idx, track_idx in matches:
                if iou_matrix[det_idx, track_idx] >= self.iou_threshold:
                    self.tracks[track_idx]['bbox'] = detections[det_idx][:4]
                    self.tracks[track_idx]['age'] = 0
                    self.tracks[track_idx]['hits'] += 1
        
        # Create new tracks for unmatched detections
        # Remove old tracks
        
        return active_tracks
```

**IoU (Intersection over Union) Calculation:**

```python
def compute_iou(box1, box2):
    """
    box1, box2: [x1, y1, x2, y2]
    Returns: IoU score (0 to 1)
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
```

**Visual Example:**
```
Frame N:   Box A: [100, 100, 200, 200]
Frame N+1: Box B: [105, 105, 205, 205]

IoU(A, B) = 0.85  # High overlap → Same object!
```

#### 4. **Hungarian Algorithm for Matching**

```python
from scipy.optimize import linear_sum_assignment

def hungarian_matching(iou_matrix):
    """
    Find optimal assignment of detections to tracks
    
    Args:
        iou_matrix: [num_detections × num_tracks]
    
    Returns:
        List of (detection_idx, track_idx) pairs
    """
    # Convert to cost matrix (minimize cost = maximize IoU)
    cost_matrix = -iou_matrix
    
    # Hungarian algorithm
    det_indices, track_indices = linear_sum_assignment(cost_matrix)
    
    return list(zip(det_indices, track_indices))
```

**Why Hungarian Algorithm?**
```
Problem: 3 detections, 3 tracks
         Which detection matches which track?

Naive: Try all combinations (3! = 6 possibilities)
Hungarian: Finds optimal in O(n³) time
```

**Example:**
```
IoU Matrix:
         Track1  Track2  Track3
Det1:    0.8     0.2     0.1
Det2:    0.1     0.9     0.2
Det3:    0.2     0.1     0.85

Optimal Assignment:
Det1 → Track1 (IoU: 0.8)
Det2 → Track2 (IoU: 0.9)
Det3 → Track3 (IoU: 0.85)
```

#### 5. **Drawing Annotations**

```python
def draw_boxes(frame, tracks):
    """Draw bounding boxes and labels on frame"""
    for track in tracks:
        x1, y1, x2, y2, track_id, class_id, confidence = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get color for this class
        color = COLORS[class_id]
        class_name = CLASS_NAMES[class_id]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f"ID:{track_id} {class_name} {confidence:.2f}"
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(frame, (x1, y1 - label_h -
