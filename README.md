# ConcreteVision

<p align="center">
  <h2 align="center">AI-Powered Semantic Segmentation Framework for Concrete Structural Damage Analysis</h2>
</p>

<p align="center">
  Deep Learning • Semantic Segmentation • Infrastructure Inspection • ONNX Deployment • Explainable AI
</p>

---

## Overview

ConcreteVision is a deep learning framework designed for semantic segmentation and analysis of concrete structural damages using computer vision techniques.

The project focuses on:
- Multi-class concrete damage segmentation
- Sliding-window inference for high-resolution images
- Test-Time Augmentation (TTA)
- ONNX model export and deployment
- Scalable deployment architecture for web and cloud systems

This repository is designed for:
- Research experimentation
- Infrastructure inspection workflows
- AI-assisted damage assessment
- Web-based deployment systems
- Cloud inference pipelines

---

## Features

- DeepLabV3+ semantic segmentation pipeline
- Transformer-based encoder support (MiT-B5)
- Multi-label segmentation support
- ONNX export support
- Overlay visualization utilities
- Evaluation metrics and analysis
- Config-driven experimentation
- Docker-ready architecture
- Modular Python package structure

---

## Repository Structure

```bash
ConcreteVision/
│
├── configs/                   # Training config
├── models/                    # model class
├── src/
│   └── damage_segmentation/
│       ├── datasets/
│       ├── models/
│       ├── training/
│       ├── inference/
│       ├── evaluation/
│       └── utils/
├── pyproject.toml
├── setup.py
├── requirements.txt
└── README.md
```

---

## Dataset Information

# dacl10k Dataset

dacl10k stands for *damage classification 10k images* and is a **multi-label semantic segmentation** dataset for 19 classes (13 damages and 6 objects) present on bridges. 

This dataset is used in the challenge associated with the **"1st Workshop on Vision-Based Structural Inspections in Civil Engineering" at [WACV2024](https://wacv2024.thecvf.com/workshops/).**


# Citation

* Link to the paper: [arXiv](https://arxiv.org/abs/2309.00460)


ConcreteVision is developed for semantic segmentation of Structural Concrete Damages using:
- DACL Challenge dataset
- Custom preprocessing pipelines
- Polygon-based annotations

### Supported Damage Categories

- Crack
- Corrosion
- Efflorescence
- Spalling
- Weathering
- Structural components like Beam, Protective Equipment
- Hollow Areas, Cavity



## Installation

### Clone Repository

```bash
git clone https://github.com/vedant-girish-dalvi/ConcreteVision.git

cd ConcreteVision
```

---

## Create Environment

### Conda

```bash
conda create -n concretevision python=3.10

conda activate concretevision
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Package Setup

ConcreteVision uses a modern Python package structure.

Install the project in editable mode:

```bash
pip install -e .
```

This enables:
- modular imports
- package execution
- IDE support
- Docker compatibility
- scalable deployment workflows

---

## Running Modules

Run modules from the project root.

### Training

```bash
python -m damage_segmentation.training.train
```

---

### Evaluation

```bash
python -m damage_segmentation.evaluation.evaluate
```

---

### Inference

```bash
python -m damage_segmentation.inference.inference
```

---

## Example Import Structure

### Recommended

```python
from damage_segmentation.datasets.dacl_dataset import DaclDataset

from damage_segmentation.utils.visualization import visualize_segmentation
```

### Avoid

```python
from utils import visualize_segmentation
```

---

## ONNX Export

```bash
python -m damage_segmentation.deployment.export_onnx
```

---

## Visualization Features

ConcreteVision supports:
- Colored segmentation masks
- Labeled Overlay visualizations
- High-resolution inference visualization

---

## Future Roadmap

- FastAPI backend
- React frontend
- AWS cloud deployment
- Real-time inference API

---

## Technology Stack

### Deep Learning

- PyTorch
- Segmentation Models PyTorch


<!---
### Backend (Planned)

- FastAPI
- Docker
- AWS

### Frontend (Planned)

- React
- Next.js
- Tailwind CS

--->
---

## Packaging and Development

ConcreteVision uses:
- `pyproject.toml`
- `setup.py`
- editable package installation (`pip install -e .`)

for:
- professional package management
- deployment readiness
- modular development
- scalable architecture


---
<!---
## Results

| Metric | Status |
|--------|--------|
| Mean IoU | In Progress |
| Dice Score | In Progress |
| Pixel Accuracy | In Progress |

---
--->

## Citation

If you use this repository in your research, please cite:

```bibtex
@misc{concretevision2026,
  title={ConcreteVision: Deep Learning Framework for Concrete Damage Segmentation},
  author={Vedant Girish Dalvi},
  year={2026},
  url={https://github.com/vedant-girish-dalvi/ConcreteVision}
}
```

---

## Repository Topics / Tags

```text
deep-learning
computer-vision
semantic-segmentation
pytorch
deeplabv3plus
transformers
infrastructure-inspection
damage-detection
construction-ai
onnx
aws
fastapi
```

<!---
---

## Recommended GitHub Topics

Add these in GitHub repository settings:

- semantic-segmentation
- computer-vision
- deep-learning
- pytorch
- deeplabv3plus
- infrastructure-inspection
- damage-detection
- onnx
- aws
- fastapi

---

## Repository Setup Commands

### Initialize Git

```bash
git init
```

### Add Remote Repository

```bash
git remote add origin https://github.com/vedant-girish-dalvi/ConcreteVision.git
```

### Push to GitHub

```bash
git add .

git commit -m "Initial commit"

git push origin main
```

---

## Recommended Development Workflow

```bash
git add .

git commit -m "your message"

git push
```

--->

---

## License

This project is licensed under the MIT License.

---

## Author

### Vedant Girish Dalvi

GitHub:
https://github.com/vedant-girish-dalvi

---

## Repository Status

Active Development

ConcreteVision is currently under active research and development with ongoing work in:
- segmentation improvements
- deployment optimization
- cloud inference systems
- web application integration





