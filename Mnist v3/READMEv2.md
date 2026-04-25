# TI with Exceptions — Parameter Reference

## Ordering

- **n**: Number of items in the transitive chain. Default: `8`. Produces ordering `[0, 1, 2, ..., 7]` where `0 > 1 > 2 > ... > 7`.
- **exception_pair**: Tuple `(p, q)` where `p` beats `q` despite `q > p` in the normal ordering. Requires `|p - q| >= 2`. Default: `(5, 2)` meaning 5 beats 2.

## Training Data (TransitiveTrainDataset / TransitiveTrainDataset_Exp)

- **samples_per_pair**: Number of unique MNIST image pairings generated per pair per epoch. Default: `2000`. Each sample produces 2 training examples (winner left + winner right), so total training examples = `num_pairs × samples_per_pair × 2`.
  - Base case: 7 adjacent pairs → 28,000 examples
  - Exception case: 8 pairs (7 adjacent + 1 exception) → 32,000 examples
- Sampling is on-the-fly in `__getitem__`, so fresh MNIST images are drawn every epoch.

## Test Data (TransitiveTestDataset)

- **samples_per_pair**: Same meaning as above. Default: `2000`.
- Tests all non-adjacent pairs (symbolic distance >= 2). With n=8 that's 21 pairs → 84,000 test examples.
- Also sampled on-the-fly.

## Model (Net)

- **conv1**: 1 → 32 channels, 3×3 kernel
- **conv2**: 32 → 64 channels, 3×3 kernel
- **max_pool**: 2×2
- **dropout1**: 0.25 (after conv layers)
- **fc1**: 19968 → 128
- **dropout2**: 0.5 (after fc1)
- **fc2**: 128 → 2 (binary: left wins vs right wins)
- Input shape: `(1, 28, 56)` — two 28×28 MNIST images concatenated horizontally

## Optimizer

- **SGD** with `lr=0.01`, `momentum=0.9`
- No learning rate scheduler

## Training Loop (Mnist_TI.py / Mnist_TI_Exp.py)

- **batch_size**: Default `64`. Set via `--batch-size`.
- **test_batch_size**: Default `1000`. Set via `--test-batch-size`.
- **epochs**: Default `14`. Set via `--epochs`. May need 100+ for small `samples_per_pair`.
- **shuffle**: `True` for train loader. Critical — without this the network sees pairs in blocks and suffers catastrophic forgetting across the shared binary head.
- **seed**: Default `1`. Set via `--seed`.
- **log_interval**: Print training loss every N batches. Default `10`.
- **save_model**: Save weights to `ti_cnn.pt` or `ti_exp_cnn.pt`. Set via `--save-model`.

## MNIST Preprocessing

- `ToTensor()` + `Normalize(mean=0.1307, std=0.3081)`

## Evaluation (evaluate_probs.py / evaluate_probs_exp.py)

- **batch_size**: Forward pass batch size for evaluation. Default `1000`.
- Outputs per-pair P(correct) averaged over all samples, split by winner position (left vs right).
- Generates heatmaps: position-invariant, position-split, and position-bias.
