import pandas as pd
import random

#Main data list to store all reviews
data = []

# Define possible topics and sentiments
topics = [
    "Driver Behavior", "Pricing", "App Experience", "Ride Comfort",
    "Customer Support", "Payment Issues", "Miscellaneous"
]
sentiments = ["Positive", "Neutral", "Negative"]


#Approach 1 (More reliable and better result : 1000 reviews)

positive_phrases = [
    "The {} was {} and {}.", "I had a {} experience with the {}.", "Everything about the {} was {}.",
    "Really impressed with the {} - it was {}.", "Had a {} time using the {} service!",
]

neutral_phrases = [
    "The {} was {} but nothing special.", "It was an {} experience overall.", 
    "Neither too good nor too bad - just {}.", "An {} ride, nothing to complain about.",
    "The {} worked as expected, {} experience."
]

negative_phrases = [
    "The {} was {} and really frustrating.", "Had a {} experience with the {}.", 
    "Disappointed with the {}, it was really {}.", "A very {} {} ruined the experience.", 
    "Would not recommend due to the {} being {}."
]

adjectives_positive = ["amazing", "smooth", "excellent", "great", "comfortable", "fast"]
adjectives_neutral = ["okay", "average", "decent", "standard", "normal", "fair"]
adjectives_negative = ["terrible", "bad", "poor", "slow", "uncomfortable", "frustrating"]


for _ in range(1000):
    topic = random.choice(topics)
    sentiment = random.choice(sentiments)

    if sentiment == "Positive":
        phrase = random.choice(positive_phrases)
        adj1, adj2 = random.choices(adjectives_positive, k=2)
    elif sentiment == "Neutral":
        phrase = random.choice(neutral_phrases)
        adj1, adj2 = random.choices(adjectives_neutral, k=2)
    else:
        phrase = random.choice(negative_phrases)
        adj1, adj2 = random.choices(adjectives_negative, k=2)

    review = phrase.format(topic, adj1, adj2)
    data.append([review, topic, sentiment])


#Approach 2 (Less Reliable ,reviews are repeated : 1000 reviews)


# Sample reviews for each topic-sentiment combination
review_samples = {
    ("Driver Behavior", "Positive"): [
        "The driver was polite and professional.", "Smooth ride with a friendly driver.",
        "Loved the service, driver was punctual and courteous."
    ],
    ("Driver Behavior", "Neutral"): [
        "Driver was okay, nothing special.", "Decent ride but lacked engagement.",
        "The driver followed the route correctly."
    ],
    ("Driver Behavior", "Negative"): [
        "Driver was rude and impatient.", "Very reckless driving, felt unsafe.",
        "Driver refused to turn on the AC."
    ],
    ("Pricing", "Positive"): [
        "Great value for money!", "The fare was reasonable for the distance.",
        "Affordable and transparent pricing."
    ],
    ("Pricing", "Neutral"): [
        "Price was fine, not too high, not too low.", "It was the expected fare.",
        "No surprises in pricing."
    ],
    ("Pricing", "Negative"): [
        "Too expensive for such a short ride!", "Surge pricing was unfair.",
        "Charged extra without explanation."
    ],
    ("App Experience", "Positive"): [
        "Easy to book a ride using the app.", "User-friendly interface, smooth booking.",
        "App worked flawlessly."
    ],
    ("App Experience", "Neutral"): [
        "App worked fine but nothing special.", "Had to restart the app once.",
        "Overall, a standard experience."
    ],
    ("App Experience", "Negative"): [
        "App kept crashing while booking.", "Difficult to navigate the app.",
        "Long loading times, frustrating experience."
    ],
    ("Ride Comfort", "Positive"): [
        "Very comfortable seats and clean car.", "Smooth and relaxing ride.",
        "Loved the premium ride experience."
    ],
    ("Ride Comfort", "Neutral"): [
        "Ride was okay, nothing extraordinary.", "Seats were neither too comfortable nor uncomfortable.",
        "Car was clean but not luxurious."
    ],
    ("Ride Comfort", "Negative"): [
        "Seats were dirty and uncomfortable.", "Ride was bumpy and unpleasant.",
        "Car smelled bad, ruined the experience."
    ],
    ("Customer Support", "Positive"): [
        "Support team resolved my issue quickly.", "Very helpful and responsive customer care.",
        "Got my refund without hassle."
    ],
    ("Customer Support", "Neutral"): [
        "Customer support was okay, took some time.", "They eventually helped, but not promptly.",
        "Standard service, nothing great."
    ],
    ("Customer Support", "Negative"): [
        "No response from customer support!", "Terrible service, issue not resolved.",
        "Long wait times and unhelpful support."
    ],
    ("Payment Issues", "Positive"): [
        "Seamless payment experience.", "Cashless transactions work perfectly.",
        "Payment was quick and hassle-free."
    ],
    ("Payment Issues", "Neutral"): [
        "Payment was processed, but took some time.", "Had to re-enter card details, minor inconvenience.",
        "Standard payment process, nothing special."
    ],
    ("Payment Issues", "Negative"): [
        "Double charged for my ride!", "Payment failed multiple times.", "Refund took too long to process."
    ],
    ("Miscellaneous", "Positive"): [
        "Overall a great experience!", "Everything was smooth and well-managed.",
        "Would definitely use this service again."
    ],
    ("Miscellaneous", "Neutral"): [
        "Just an average experience.", "Nothing remarkable, but it worked.",
        "Neither good nor bad, just okay."
    ],
    ("Miscellaneous", "Negative"): [
        "Not satisfied with the overall service.", "So many issues, wouldn't recommend.",
        "Worst experience I've had with a cab service."
    ]
}

# Generate dataset with 1000 reviews
for _ in range(1000):
    topic = random.choice(topics)
    sentiment = random.choice(sentiments)
    review = random.choice(review_samples[(topic, sentiment)])
    data.append([review, topic, sentiment])


# Create DataFrame and save
df = pd.DataFrame(data, columns=["Review", "Topic", "Sentiment"])
df.to_csv("Dataset\training_labeled_cab_reviews.csv", index=False)
df.head()