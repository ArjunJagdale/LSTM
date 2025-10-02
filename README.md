---
title: Twitter Sentiment Analysis with PyTorch
emoji: 💬
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.17.1
app_file: app.py
pinned: false
license: mit
short_description: Real-Time Sentiment Analysis on Airline Tweets using LSTM in PyTorch
---

## 🚀 **Overview**

This project performs **real-time sentiment analysis** on airline-related tweets using a custom-built **LSTM model** in **PyTorch**. It classifies tweets as **positive** or **negative**, completely excluding neutral sentiments for a more opinion-focused analysis.


## Demo

https://github.com/user-attachments/assets/521692ed-e979-47df-a52c-88969d3028ce

---

## 🧩 **Key Features**

- 🔎 **Text Sentiment Detection**: Predicts whether a tweet is **positive** or **negative**.
- ⚡ **Real-Time Interface**: Built using **Gradio** for quick and intuitive interaction.
- 🧠 **Custom LSTM Architecture**: A lightweight, efficient model built from scratch using **PyTorch**.
- 🗂️ **Clean Data Pipeline**: Manual vocabulary creation, tokenization, and dynamic padding without external NLP libraries.
- 📦 **Deployable Anywhere**: Lightweight, local, and Hugging Face-compatible app.

---

## 🛠️ **Tech Stack**

- **Deep Learning**: PyTorch  
- **Data Handling**: Pandas  
- **UI Interface**: Gradio  
- **Model Type**: LSTM for binary sentiment classification  
- **Tokenizer**: Custom word-level tokenizer  
- **Inference**: Binary output (positive/negative) via Sigmoid

---

## 🧠 **How It Works**

1. **Dataset**: Loads the `"Tweets.csv"` file and filters out all **neutral** sentiments.
2. **Preprocessing**:
   - Lowercases and splits each tweet into tokens.
   - Builds a custom vocabulary (`<PAD>` and `<UNK>` included).
   - Converts text into indexed tensors.
3. **Model**:
   - Uses an `Embedding` layer followed by an `LSTM` and a `Linear` layer with `Sigmoid` activation.
   - Optimized using `BCELoss` and `Adam` optimizer.
4. **Training**:
   - Trained over 10 epochs with batch-wise padding via `collate_fn`.
   - Tracks and prints loss and accuracy for each epoch.
5. **Inference**:
   - Text input is tokenized, encoded, and passed through the model for prediction.
   - Displays `positive` or `negative` based on threshold (0.5).

---

## 📦 **File Structure**

```plaintext
sentiment-analysis-app/
│
├── app.py               # Main application script
├── Tweets.csv           # Dataset file
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
