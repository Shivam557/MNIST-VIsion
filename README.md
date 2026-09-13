# MNIST-Vision

PyTorch implementation of a handwritten digit classifier trained on the MNIST dataset.

## Installation

```bash
git clone https://github.com/Shivam557/MNIST-VIsion.git
cd MNIST-VIsion
pip install -r requirements.txt
```

## Dataset

MNIST contains **70,000 grayscale images** of handwritten digits from `0` to `9`.

* Image size: `28 × 28`
* Classes: `10`
* Training images: `60,000`
* Test images: `10,000`

## Usage

### Train

```bash
python train.py
```

### Predict

```bash
python predict.py
```

## Model

The classifier is implemented directly with PyTorch.

```text
Input (28 × 28)
      ↓
Flatten
      ↓
Linear
      ↓
ReLU
      ↓
Linear
      ↓
ReLU
      ↓
Linear
      ↓
Output (10 classes)
```

## Results

| Metric          | Value |
| --------------- | ----: |
| Test Accuracy   |     — |
| Training Epochs |     — |
| Parameters      |     — |

Results will be updated after training.

## Project Structure

```text
MNIST-Vision/
├── train.py
├── predict.py
├── model.py
├── requirements.txt
└── README.md
```

## Stack

* Python
* PyTorch
* torchvision

## License

MIT
