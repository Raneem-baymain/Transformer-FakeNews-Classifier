[README.md](https://github.com/user-attachments/files/28810742/README.md)
# Milestone 1A — Transformer Encoder Implementation

## Objective

Implement the core components of a Transformer encoder from scratch and verify correctness using a small subset of the dataset.

## Implemented Components

- Scaled Dot-Product Attention
- Multi-Head Attention (custom implementation)
- Transformer Encoder Block
- Learned Positional Embedding
- Transformer Classifier (classification head with mean pooling)

## Files

| File | Description |
| --- | --- |
| `Multihead__milestone1t_fixed.ipynb` | Main notebook: model implementation, unit tests, preprocessing, and dry run |
| `config.py` | All hyperparameters and the global random seed |
| `README.md` | This file |

## How to Run

1. Open `Multihead__milestone1t_fixed.ipynb` in Google Colab (GPU runtime recommended).
2. Run all cells top-to-bottom (**Runtime → Run all**). The notebook will:
   - clone this repository and load `config.py`,
   - run the correctness unit tests (shape, mask, attention-sum, gradient flow),
   - download the WELFake dataset via `kagglehub`,
   - preprocess, split (80/10/10), and encode the data,
   - execute the dry run (5,000 train / 500 val samples, 5 epochs) and report metrics.

## Reproducibility

All experiments use a fixed random seed (`SEED = 42`, defined in `config.py`),
applied at the top of the notebook **before any data loading or model creation**:

- **Python** `random.seed(SEED)`
- **NumPy** `np.random.seed(SEED)`
- **PyTorch (CPU & CUDA)** `torch.manual_seed(SEED)` / `torch.cuda.manual_seed_all(SEED)`
- **cuDNN** `deterministic = True`, `benchmark = False`

The seed also covers every random operation downstream:

| Source of randomness | How it is seeded |
| --- | --- |
| Train/val/test split | `train_test_split(..., random_state=config.SEED)` |
| DataLoader shuffling | `DataLoader(..., generator=torch.Generator().manual_seed(SEED))` |
| Weight initialization | global `torch.manual_seed(SEED)` |
| Dropout | global `torch.manual_seed(SEED)` |

Re-running the notebook top-to-bottom therefore reproduces the reported losses,
accuracy, and confusion matrix exactly.

**Verification:** two consecutive seeded runs of the dry-run training loop produce
identical per-epoch losses, while unseeded runs diverge from the first epoch.

*Note:* exact bit-level results are guaranteed on the same hardware/software stack
(e.g. the same Colab GPU runtime). Across different GPU models, minor floating-point
differences in CUDA kernels can cause negligible variations.

## Evaluation Convention

In this task, **Fake (0) is treated as the positive class**, since the system's
objective is detecting fake news. Accordingly, in the confusion-matrix breakdown:

- **False Negative** = a fake article predicted as real (a missed fake — the critical error)
- **False Positive** = a real article predicted as fake (a false alarm)
