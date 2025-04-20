# Text Classification Models

This repository contains two different approaches for text classification:

1. A fine-tuned DistilBERT model using Hugging Face Transformers
2. A traditional Naive Bayes classifier with TF-IDF features

## Model 1: Fine-tuned DistilBERT (`distilbert_finetuned_model.py`)

A state-of-the-art transformer-based model fine-tuned for sequence classification tasks.

### Features:
- Uses the `distilbert-base-uncased` pre-trained model
- Implements dynamic padding and truncation
- Supports multi-class classification
- Includes evaluation metrics (accuracy and F1-score)
- Saves and loads trained models for reuse
- Automatic mixed-precision training when GPU is available

### Usage:
```python
from distilbert_finetuned_model import Model

# Initialize and train
model = Model()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

### Training Parameters:
- Learning rate: 2e-5
- Batch size: 8
- Epochs: 3
- Weight decay: 0.01

## Model 2: Naive Bayes with TF-IDF (`NB_model.py`)

A traditional machine learning approach for text classification.

### Features:
- Uses TF-IDF vectorization with English stop words removal
- Implements Multinomial Naive Bayes classifier
- Handles up to 10,000 features
- Includes label encoding for text labels

### Usage:
```python
from NB_model import Model

# Initialize and train
model = Model()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

## Comparison

| Feature                | DistilBERT Model | Naive Bayes Model |
|------------------------|------------------|-------------------|
| Architecture           | Transformer      | Traditional ML    |
| Pretrained             | Yes              | No                |
| Computational Needs    | High (GPU recommended) | Low       |
| Training Time          | Longer           | Faster            |
| Typical Accuracy       | Higher           | Lower             |
| Feature Engineering    | Not needed       | TF-IDF required   |

## Requirements

- Python 3.6+
- PyTorch
- Transformers library
- Scikit-learn
- Datasets library (Hugging Face)
- NumPy
- Pandas

## Installation

```bash
pip install torch transformers scikit-learn datasets numpy pandas
```

## Which to Choose?

- Use the **DistilBERT model** when you need state-of-the-art accuracy and have access to GPU resources
- Use the **Naive Bayes model** when you need fast training/prediction or are working with limited computational resources

Both models implement the same interface (`fit()` and `predict()` methods) making it easy to switch between them.
