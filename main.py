import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Define topics and sentiments
TOPICS = ["Driver Behavior", "Pricing", "App Experience", "Ride Comfort", "Customer Support", "Payment Issues", "Miscellaneous"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]

# Load the tokenizer and saved models
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_topic = BertForSequenceClassification.from_pretrained("TrainedClassifierModels/./topic_classifier")
model_sentiment = BertForSequenceClassification.from_pretrained("TrainedClassifierModels/./sentiment_classifier")

# Load new data
new_df = pd.read_csv("Dataset/testing_uber_reviews.csv")  # Ensure correct path
new_df.dropna(subset=["content"], inplace=True)  # Remove missing values

# Function to predict topic & sentiment
def predict_topic_sentiment(text):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    
    with torch.no_grad():  # No need for gradient computation
        topic_logits = model_topic(**encoding).logits.numpy()
        sentiment_logits = model_sentiment(**encoding).logits.numpy()
    
    topic = TOPICS[np.argmax(topic_logits)]
    sentiment = SENTIMENTS[np.argmax(sentiment_logits)]
    
    return topic, sentiment

# Apply prediction to all reviews
new_df[['Predicted_Topic', 'Predicted_Sentiment']] = new_df['content'].apply(lambda x: pd.Series(predict_topic_sentiment(x)))

# Save results
new_df.to_csv("Results/predicted_reviews.csv", index=False)
print("Predictions saved to predicted_reviews.csv!")
