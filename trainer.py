import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Define topics and sentiments
TOPICS = ["Driver Behavior", "Pricing", "App Experience", "Ride Comfort", "Customer Support", "Payment Issues", "Miscellaneous"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]

# Load dataset
df = pd.read_csv("Dataset\labeled_cab_reviews.csv")  

# Remove missing values
df.dropna(subset=["Review", "Topic", "Sentiment"], inplace=True)
df = df[df["Topic"].isin(TOPICS)]


# Encode labels
df['topic_label'] = df['Topic'].apply(lambda x: TOPICS.index(x))
df['sentiment_label'] = df['Sentiment'].apply(lambda x: SENTIMENTS.index(x))

# Split dataset
train_texts, val_texts, train_topic_labels, val_topic_labels, train_sent_labels, val_sent_labels = train_test_split(
    df['Review'].tolist(), df['topic_label'].tolist(), df['sentiment_label'].tolist(), test_size=0.2, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Custom Dataset Class
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create dataset objects
train_topic_dataset = ReviewDataset(train_texts, train_topic_labels, tokenizer)
val_topic_dataset = ReviewDataset(val_texts, val_topic_labels, tokenizer)
train_sent_dataset = ReviewDataset(train_texts, train_sent_labels, tokenizer)
val_sent_dataset = ReviewDataset(val_texts, val_sent_labels, tokenizer)

# Load pre-trained BERT models
model_topic = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7)
model_sentiment = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Compute Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

# Training Arguments
training_args = TrainingArguments(
    output_dir="./result",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Train Topic Model
trainer_topic = Trainer(
    model=model_topic,
    args=training_args,
    train_dataset=train_topic_dataset,
    eval_dataset=val_topic_dataset,
    compute_metrics=compute_metrics
)
trainer_topic.train()
model_topic.save_pretrained("TrainedClassifierModels/./topic_classifier")

# Train Sentiment Model
trainer_sentiment = Trainer(
    model=model_sentiment,
    args=training_args,
    train_dataset=train_sent_dataset,
    eval_dataset=val_sent_dataset,
    compute_metrics=compute_metrics
)
trainer_sentiment.train()
model_sentiment.save_pretrained("TrainedClassifierModels/./sentiment_classifier")

# Prediction on New Data
def predict_topic_sentiment(text):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    
    topic_logits = model_topic(**encoding).logits.detach().numpy()
    sentiment_logits = model_sentiment(**encoding).logits.detach().numpy()
    
    topic = TOPICS[np.argmax(topic_logits)]
    sentiment = SENTIMENTS[np.argmax(sentiment_logits)]
    
    return topic, sentiment

# Example Prediction
new_review = "The driver was very polite and helpful."
pred_topic, pred_sentiment = predict_topic_sentiment(new_review)
print(f"Predicted Topic: {pred_topic}, Predicted Sentiment: {pred_sentiment}")

print("Training complete! Models saved.")