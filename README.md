# 🚖 Cab Review Classifier (Topic + Sentiment Analysis)

This project is an end-to-end NLP pipeline that classifies cab/ride-sharing reviews into **topics** and their corresponding **sentiments** using BERT.

---

## 📚 Problem Statement

Given user reviews from cab services like Uber, we aim to:
1. Classify each review into one of the following **topics**:
   - Driver Behavior
   - Pricing
   - App Experience
   - Ride Comfort
   - Customer Support
   - Payment Issues
   - Miscellaneous
2. Then determine the **sentiment** of that review:
   - Positive, Neutral, or Negative

---

## 🛠️ Project Workflow

1. **`dataset_creation.py`**  
   - Generates synthetic labeled data using Python’s `random` library.
   - Used because no open-source, labeled Uber review dataset exists.

2. **`trainer.py`**
   - Fine-tunes BERT models separately for:
     - **Topic classification**
     - **Sentiment analysis**
   - Saves two trained models locally.

3. **`main.py`**
   - Loads the trained models.
   - Takes in raw user reviews and outputs predicted topic and sentiment.

---

## 📁 Files & Directory Structure

cab-review-classifier/ ├── dataset_creation.py # Synthetic dataset generator ├── trainer.py # BERT fine-tuning script ├── main.py # Inference ├── requirements.txt # Dependencies ├── README.md # This file └── TrainedClassifierModels/ # (Not uploaded) Contains trained BERT models


