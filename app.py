import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import gradio as gr

# Load dataset
df = pd.read_csv('Tweets.csv')[['text', 'airline_sentiment']]
df = df[df['airline_sentiment'] != 'neutral']

# Label encoding
label_map = {'positive': 1, 'negative': 0}
df['airline_sentiment'] = df['airline_sentiment'].map(label_map)

# Tokenization
def tokenize(text):
    return text.lower().split()

# Build vocabulary
unique_tokens = set(token for text in df['text'] for token in tokenize(text))
vocab = {word: idx + 2 for idx, word in enumerate(unique_tokens)}  # Reserve 0 for PAD, 1 for UNK
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode_text(text):
    return torch.tensor([vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)], dtype=torch.long)

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [encode_text(text) for text in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Padding function
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=vocab["<PAD>"])
    return texts, torch.tensor(labels, dtype=torch.float32)

# Prepare dataset
dataset = SentimentDataset(df['text'].tolist(), df['airline_sentiment'].tolist())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# Define model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return self.sigmoid(x)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
model = SentimentLSTM(vocab_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += ((outputs > 0.5).float() == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {correct/len(dataset):.4f}")

# Prediction function
def predict_sentiment(text):
    model.eval()
    tokens = encode_text(text).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tokens).item()
    return "positive" if prediction > 0.5 else "negative"

# Gradio Interface with examples
examples = [
    ["The check-in process was smooth. The staff was friendly and helpful. The flight landed on time."],
    ["The flight was delayed for 3 hours. The customer service was unresponsive. The food was terrible."],
    ["The meal was good, but the entertainment system was not working. The seats were a bit cramped."],
]

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter your review here...", label="Input Review"),
    outputs="text",
    title="Real-Time Sentiment Analysis using LSTM",
    examples=examples
)
iface.launch()
