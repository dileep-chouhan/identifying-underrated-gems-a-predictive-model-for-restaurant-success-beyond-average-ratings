import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
# Generate synthetic data for restaurants
num_restaurants = 100
data = {
    'Restaurant': [f'Restaurant {i+1}' for i in range(num_restaurants)],
    'Average_Rating': np.random.uniform(2.5, 4.5, num_restaurants),
    'Review_Text': [f'This restaurant is {np.random.choice(["amazing", "good", "okay", "bad", "terrible"])}. The food was {np.random.choice(["delicious", "tasty", "average", "bland", "inedible"])}.' for _ in range(num_restaurants)]
}
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
df['Sentiment_Score'] = df['Review_Text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])
# --- 3. Feature Engineering ---
#  A simple approach:  Restaurants with higher sentiment scores but lower average ratings are potential "underrated gems"
df['Underrated_Potential'] = df['Sentiment_Score'] - df['Average_Rating']
# --- 4. Analysis ---
# Identify top 10 restaurants with high underrated potential
top_underrated = df.nlargest(10, 'Underrated_Potential')
print("Top 10 Underrated Restaurants:")
print(top_underrated[['Restaurant', 'Average_Rating', 'Sentiment_Score', 'Underrated_Potential']])
# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average_Rating', y='Sentiment_Score', hue='Underrated_Potential', size='Underrated_Potential', data=df)
plt.title('Restaurant Sentiment vs. Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'restaurant_sentiment_vs_rating.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Further analysis could involve more sophisticated NLP techniques, regression modeling etc.  This is a basic example.