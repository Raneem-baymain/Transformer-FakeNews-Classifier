# Milestone 1A — Transformer Encoder Implementation

## Objective
Implement the core components of a Transformer encoder from scratch and verify correctness using a small subset of the dataset.

## Implemented Components
* **Scaled Dot-Product Attention**
* **Multi-Head Attention** (custom implementation)
* **Transformer Encoder Block**
* **Position-wise Feed Forward Network (FFN)**
* **Learned Positional Embedding**
* **Transformer-based Classifier**

---

## Run Instructions

### Step 1: Navigate to Folder
```bash
cd milestone1A
```

### Step 2: Run the Code

#### Option A — Jupyter Notebook (Recommended)
```bash
jupyter notebook
```
Open `multihead__milestone1t.ipynb` and run all cells sequentially.

#### Option B — Python Script
```bash
python multihead__milestone1t.py
```

---

## Pipeline

### 1. Data Processing
* Load a small subset of the dataset.
* Apply preprocessing: text cleaning, tokenization, and vocabulary construction.
* Convert tokens to numerical indices.
* Apply padding and truncation.

### 2. Model Forward Pass
* Input -> Embedding layer
* Add positional embeddings
* Pass through Transformer encoder block:
  * Multi-Head Attention
  * Feed Forward Network
* Apply mean pooling
* Pass through classification layer

### 3. Training (Subset Validation)
* **Loss:** `CrossEntropyLoss`
* **Optimizer:** `Adam`
* Small number of iterations for validation

---

## Validation Checks
The implementation is verified using:
* **Tensor shape checks** across all layers.
* **Attention weight normalization** (sum ≈ 1 along correct dimension).
* **Mask correctness** (padding behavior).
* **Gradient flow validation** (loss decreases on small batch).

## Expected Output
* Training loss values.
* Correct tensor shape logs.
* Attention weights behaving as expected.
* Successful forward and backward pass.

---

> ### Notes
> * This milestone focuses on correctness, not performance.
> * Uses a small subset to ensure fast debugging.
> * No built-in Transformer modules are used (implemented from scratch).
> * Serves as the foundation for Milestone 1B.
