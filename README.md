# Convolutional Neural Networks — Fashion-MNIST

## Problem Description

This assignment explores convolutional layers as an **architectural decision**
rather than a black-box recipe: they encode an inductive bias (locality +
translation invariance) directly into the model. The goal is to design,
justify, and experimentally evaluate a CNN architecture against a
non-convolutional baseline on real image data, and to reason about *why* the
architectural choice matters — not just to hit a target accuracy.

## Dataset Description

**Fashion-MNIST** (Zalando Research), loaded via `tf.keras.datasets.fashion_mnist`.

- 60,000 train / 10,000 test grayscale images, 28×28, 1 channel
- 10 balanced classes (6,000 train / 1,000 test images each): T-shirt/top,
  Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- Chosen because it is image-based, grid-structured, small enough to train
  quickly, and harder than digit-MNIST (some classes — Shirt, T-shirt/top,
  Pullover, Coat — are visually close and differ mainly in local
  texture/edges), which makes the CNN-vs-dense comparison meaningful.
- See the full justification in Section 1 of the notebook.

## Architecture Diagram

```
Baseline (non-convolutional)
  Input (28,28,1)
    -> Flatten (784)
    -> Dense(128, relu)
    -> Dense(64, relu)
    -> Dense(10, softmax)
  109,386 parameters

CNN (final architecture, 3x3 kernels)
  Input (28,28,1)
    -> Conv2D(32, 3x3, stride 1, padding=same, relu)
    -> MaxPool(2x2)                              -> (14,14,32)
    -> Conv2D(64, 3x3, stride 1, padding=same, relu)
    -> MaxPool(2x2)                              -> (7,7,64)
    -> Flatten (3136)
    -> Dense(64, relu)
    -> Dropout(0.3)
    -> Dense(10, softmax)
  220,234 parameters
```

Design justification for every choice (kernel size, stride, padding, filters,
activation, pooling, regularization) is documented in Section 4 of the
notebook.

## Experimental Results

Controlled experiment: **kernel size**, 3×3 vs. 5×5, with depth, filters,
stride, padding, pooling, optimizer, batch size and epochs held fixed.

*Run the notebook on the real dataset and paste your actual numbers here* —
the table below is the format used by the notebook's comparison cells:

| Model            | Test Accuracy | Test Loss | Parameters | Train time (s) |
|------------------|:---:|:---:|:---:|:---:|
| Baseline (Dense) |     |     | 109,386 |     |
| CNN (3×3)        |     |     | 220,234 |     |
| CNN (5×5)        |     |     | 253,514 |     |

## Interpretation

Full architectural reasoning (why convolution helps, what inductive bias it
introduces, when it would *not* be appropriate) is in Section 6 of the
notebook. Summary:

- Weight sharing across spatial positions means the CNN needs to learn far
  fewer distinct patterns than the dense baseline to cover the same visual
  variability, which shows up as better accuracy-per-parameter and less
  overfitting.
- Convolution encodes **locality** (nearby pixels matter more than distant
  ones) and **translation equivariance** (a feature is recognized the same
  way regardless of where it appears) — both correct assumptions for natural
  images.
- Convolution is a poor fit for tabular/non-grid data, permutation-invariant
  sets, graph-structured data, and problems dominated by long-range
  dependencies that span the whole input (where attention-based architectures
  are usually preferred).

## SageMaker Deployment

Section 7 of the notebook trains the final CNN with a SageMaker **TensorFlow
script-mode estimator** (`train.py`, included in the same folder) and deploys
it to a real-time inference endpoint.

Steps to run in an AWS Academy Learner Lab SageMaker notebook instance:

1. Start the Learner Lab and open a SageMaker notebook instance / Studio.
2. Upload `cnn_fashion_mnist.ipynb` and `train.py`.
3. Run through the notebook up to Section 7; it will:
   - Save the preprocessed train/test arrays as `.npy` files.
   - Upload them to the default SageMaker S3 bucket.
   - Launch a `TensorFlow` estimator training job (`ml.m5.large`, script mode).
   - Deploy the trained model to a real-time endpoint.
   - Send a handful of test images to the endpoint to sanity-check predictions.
4. **Delete the endpoint** at the end of the notebook to avoid Learner Lab
   budget consumption.
5. Take the required screenshots (training job status "Completed", endpoint
   "InService", and the prediction sanity-check output) and paste them below.

### Screenshots (to be added after running in AWS Academy)

- [ ] Training job — status "Completed"
- [ ] Endpoint — status "InService"
- [ ] Sample predictions from the deployed endpoint

## Files

- `cnn_fashion_mnist.ipynb` — full notebook (EDA, baseline, CNN, experiment, interpretation, SageMaker deployment)
- `train.py` — SageMaker script-mode training entry point (also embedded via `%%writefile` in the notebook)
- `README.md` — this file
