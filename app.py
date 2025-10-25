import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import re
from tqdm import tqdm

print("Loading IMDB dataset...")
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'<br />', ' ', text)  # Remove HTML breaks
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def tokenize(text):
    """Simple tokenization"""
    return clean_text(text).split()

print("\nBuilding vocabulary...")
# Build vocabulary from training data only
all_tokens = []
for text in tqdm(train_data['text'][:5000], desc="Processing texts"):  # Use subset for faster vocab building
    all_tokens.extend(tokenize(text))

token_counts = Counter(all_tokens)
# Keep tokens that appear at least 5 times
vocab_tokens = [word for word, count in token_counts.most_common() if count >= 5]
vocab_tokens = vocab_tokens[:10000]  # Keep top 10k words

# Create vocabulary
vocab = {"<PAD>": 0, "<UNK>": 1}
vocab.update({word: idx + 2 for idx, word in enumerate(vocab_tokens)})

print(f"Vocabulary size: {len(vocab)}")

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and encode
        tokens = tokenize(text)[:self.max_len]
        encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Pad or truncate
        if len(encoded) < self.max_len:
            encoded = encoded + [self.vocab["<PAD>"]] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]

        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

# Create datasets
print("\nCreating datasets...")
train_dataset = IMDBDataset(train_data['text'], train_data['label'], vocab)
test_dataset = IMDBDataset(test_data['text'], test_data['label'], vocab)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super(SentimentLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate last hidden states from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)

        output = self.fc(hidden)
        return output.squeeze()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = SentimentLSTM(len(vocab)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nModel architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for texts, labels in tqdm(dataloader, desc="Training"):
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Evaluating"):
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total

print("\n" + "="*60)
print("TRAINING START")
print("="*60)

num_epochs = 5
best_test_acc = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 60)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc*100:.2f}%")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'best_sentiment_model.pth')
        print(f"✓ New best model saved! (Acc: {test_acc*100:.2f}%)")

print("\n" + "="*60)
print(f"TRAINING COMPLETE - Best Test Accuracy: {best_test_acc*100:.2f}%")
print("="*60)

def predict_sentiment(text, model, vocab, device, max_len=256):
    """Predict sentiment for a given text"""
    model.eval()

    # Preprocess
    tokens = tokenize(text)[:max_len]
    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    # Pad
    if len(encoded) < max_len:
        encoded = encoded + [vocab["<PAD>"]] * (max_len - len(encoded))

    # Convert to tensor
    text_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(text_tensor)
        probability = torch.sigmoid(output).item()

    sentiment = "Positive" if probability > 0.5 else "Negative"
    confidence = probability if probability > 0.5 else 1 - probability

    return sentiment, confidence, probability

# Print detailed parameter breakdown
print("\nDetailed Model Parameters:")
print("="*60)
for name, param in model.named_parameters():
    print(f"{name:30s} | Shape: {str(list(param.shape)):25s} | Params: {param.numel():,}")
print("="*60)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# 1. Final epoch training metrics
print(f"\nFinal Training Metrics:")
print(f"Train Loss: {train_loss:.4f}")
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# 2. Training time (add at start and end of training)
import time
start_time = time.time()
# ... your training loop ...
end_time = time.time()
training_time = end_time - start_time
print(f"\nTotal training time: {training_time/60:.2f} minutes")

# 3. Test a few sample predictions
samples = [
    "This movie was absolutely fantastic!",
    "Terrible waste of time and money.",
    "It was okay, nothing special."
]
print("\nSample Predictions:")
for text in samples:
    sentiment, confidence, prob = predict_sentiment(text, model, vocab, device)
    print(f"Text: {text[:50]}")
    print(f"  → {sentiment} ({confidence*100:.1f}%)\n")

print("\nModel Architecture:")
print(model)

print("SAMPLE PREDICTIONS")
print("="*60)

test_reviews = [
    "This movie was absolutely amazing! Best film I've seen all year.",
    "Terrible waste of time. The plot made no sense and acting was horrible.",
    "It was okay, nothing special but not terrible either.",
    "A masterpiece of cinema. Beautiful cinematography and compelling story.",
    "I fell asleep halfway through. Boring and predictable.",
    "The tables were too short. The food was best!"
]

for review in test_reviews:
    sentiment, confidence, prob = predict_sentiment(review, model, vocab, device)
    print(f"\nReview: {review[:60]}...")
    print(f"Sentiment: {sentiment} (Confidence: {confidence*100:.1f}%)")

import gradio as gr

def analyze_sentiment(text):
    """Analyze sentiment and return probabilities"""
    if not text.strip():
        return {"Negative": 0.5, "Positive": 0.5}

    sentiment, confidence, prob = predict_sentiment(text, model, vocab, device)

    # Return probabilities for both classes
    return {
        "Negative": 1 - prob,
        "Positive": prob
    }

# Example reviews
examples = [
    ["This movie was absolutely fantastic! The acting was superb and I loved every minute of it."],
    ["Worst movie ever. Complete waste of time and money. Terrible acting and boring plot."],
    ["It was okay. Some good parts, some bad parts. Nothing particularly memorable."],
    ["A masterpiece! Beautiful cinematography, amazing soundtrack, and powerful performances."],
    ["I fell asleep halfway through. Slow, boring, and predictable storyline."]
]

# Create Gradio interface
print("\n" + "="*60)
print("LAUNCHING GRADIO INTERFACE")
print("="*60)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 IMDB Movie Review Sentiment Analyzer")
    gr.Markdown("Enter a movie review to analyze its sentiment using our LSTM model trained on 25,000 IMDB reviews.")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Movie Review",
                placeholder="Enter your movie review here...",
                lines=5
            )
            submit_btn = gr.Button("Analyze Sentiment", variant="primary", size="lg")

        with gr.Column():
            output = gr.Label(
                label="Sentiment Analysis",
                num_top_classes=2
            )

    gr.Examples(
        examples=examples,
        inputs=text_input,
        outputs=output,
        fn=analyze_sentiment,
        cache_examples=False
    )

    submit_btn.click(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=output
    )

    text_input.submit(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=output
    )

demo.launch(share=False)
