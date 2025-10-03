# IMDB Sentiment Analysis with PyTorch

A binary sentiment classification system for movie reviews using Bidirectional LSTM networks. Trained on the IMDB dataset with **85.19% test accuracy**.

## Overview

This project implements a deep learning model to classify movie reviews as positive or negative using PyTorch. The model uses a 2-layer Bidirectional LSTM architecture with word embeddings.

## Demo
> ScreenShot 1 of negative sentiment- 
<img width="1241" height="401" alt="image" src="https://github.com/user-attachments/assets/6912d4ed-5c54-4762-9a48-206947278e3e" />

---
> ScreenShot 2 of positive sentiment - 
<img width="1236" height="395" alt="image" src="https://github.com/user-attachments/assets/c4cc8297-8b22-4809-87ee-88debece8823" />
---

> ScreenSHot 3 of a generalized/neutral statement -
<img width="1235" height="396" alt="image" src="https://github.com/user-attachments/assets/bf93d56d-9b1f-4a9d-96f5-f11935c35b19" />




## Dataset

- **Source**: IMDB Movie Reviews (via HuggingFace `datasets`)
- **Training samples**: 25,000
- **Test samples**: 25,000
- **Classes**: Binary (0 = Negative, 1 = Positive)
- **Balance**: 50/50 split (perfectly balanced)

## Model Architecture

```
SentimentLSTM(
  (embedding): Embedding(10002, 128, padding_idx=0)
  (lstm): LSTM(128, 256, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
  (fc): Linear(in_features=512, out_features=1, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
)
```

### Architecture Details

| Component | Configuration |
|-----------|--------------|
| Vocabulary Size | 10,002 words |
| Embedding Dimension | 128 |
| LSTM Hidden Size | 256 |
| LSTM Layers | 2 (bidirectional) |
| Dropout | 0.5 |
| Output | Binary classification (1 neuron) |
| **Total Parameters** | **3,648,257** |

### Parameter Breakdown

| Layer | Parameters |
|-------|------------|
| Embedding | 1,280,256 |
| LSTM Layer 1 (Bidirectional) | 790,528 |
| LSTM Layer 2 (Bidirectional) | 1,576,960 |
| Fully Connected | 513 |
| **Total** | **3,648,257** |

## Performance

### Final Results

| Metric | Train | Test |
|--------|-------|------|
| **Accuracy** | 88.84% | **85.19%** |
| **Loss** | 0.2845 | 0.3822 |

### Sample Predictions

| Review | Prediction | Confidence |
|--------|------------|------------|
| "This movie was absolutely fantastic!" | Positive | 84.5% |
| "Terrible waste of time and money." | Negative | 95.4% |
| "It was okay, nothing special." | Negative | 81.4% |

## Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | BCEWithLogitsLoss |
| Gradient Clipping | 1.0 |
| Max Sequence Length | 256 tokens |

### Data Preprocessing

1. **Text Cleaning**
   - Convert to lowercase
   - Remove HTML tags (`<br />`)
   - Remove special characters
   - Keep only alphabetic characters

2. **Tokenization**
   - Simple whitespace tokenization
   - Vocabulary limited to top 10,000 words
   - Minimum word frequency: 5 occurrences

3. **Sequence Processing**
   - Maximum length: 256 tokens
   - Padding with `<PAD>` token (index 0)
   - Unknown words mapped to `<UNK>` token (index 1)

## Installation

```bash
pip install torch datasets gradio pandas numpy tqdm
```

## Usage

### Training

```python
python imdb_sentiment_analysis.py
```

The script will:
1. Load and preprocess the IMDB dataset
2. Build vocabulary from training data
3. Train the model for 5 epochs
4. Save the best model checkpoint
5. Launch Gradio interface for inference

### Inference with Gradio

After training, a web interface launches automatically for real-time predictions:

```python
# Interface features:
# - Text input for custom reviews
# - Probability bars for Positive/Negative sentiment
# - Pre-loaded example reviews
```

### Programmatic Inference

```python
from models import predict_sentiment

text = "Amazing movie with great acting!"
sentiment, confidence, probability = predict_sentiment(text, model, vocab, device)

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence*100:.1f}%")
```

## Technical Implementation

### Model Features

- **Bidirectional LSTM**: Processes sequences in both forward and backward directions
- **Stacked Architecture**: 2 LSTM layers for hierarchical feature learning
- **Dropout Regularization**: 50% dropout to prevent overfitting
- **Gradient Clipping**: Prevents exploding gradients (max norm = 1.0)
- **Padding Mask**: Embedding layer ignores padding tokens

### Key Components

```python
# Loss function with logits for numerical stability
criterion = nn.BCEWithLogitsLoss()

# Gradient clipping during training
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Prediction threshold
prediction = torch.sigmoid(output) > 0.5
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- HuggingFace Datasets
- Gradio 4.0+
- CUDA (optional, for GPU acceleration)

## Results Analysis

### Model Generalization

- **Train-Test Gap**: 3.65% (88.84% → 85.19%)
- **Interpretation**: Small gap indicates good generalization with minimal overfitting
- **Loss Difference**: 0.0977 (acceptable for this architecture)

### Strengths

- High confidence on clearly positive/negative reviews (>90%)
- Handles negations and sentiment modifiers effectively
- Fast inference (<100ms per review)

### Limitations

- Lower confidence on neutral/mixed reviews (~80%)
- Vocabulary limited to 10K words (OOV words mapped to `<UNK>`)
- Binary classification only (no sentiment intensity)

## Future Improvements

- Implement attention mechanism for interpretability
- Increase vocabulary size or use subword tokenization
- Add sentiment intensity scoring (0-1 scale)
- Fine-tune pre-trained embeddings (GloVe, Word2Vec)
- Experiment with Transformer architectures

## References

1. Maas, A. L., et al. (2011). "Learning Word Vectors for Sentiment Analysis." ACL.
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." Neural Computation.

## License

MIT License

---

**Model Checkpoint**: `best_sentiment_model.pth`  
**Framework**: PyTorch 2.0+  
**Accuracy**: 85.19% on IMDB test set
