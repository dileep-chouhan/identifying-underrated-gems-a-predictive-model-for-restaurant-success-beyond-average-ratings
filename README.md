# Identifying Underrated Gems: A Predictive Model for Restaurant Success Beyond Average Ratings

## Overview

This project aims to develop a predictive model that identifies restaurants with high growth potential, going beyond simple average rating scores.  Instead of relying solely on numerical ratings, this model leverages qualitative data from customer reviews to uncover hidden gems and predict future success.  The analysis involves natural language processing techniques to extract sentiment and key features from reviews, which are then used to train and evaluate a predictive model.

## Technologies Used

* Python 3
* Pandas
* NumPy
* Scikit-learn
* NLTK (Natural Language Toolkit)
* Matplotlib
* Seaborn

## How to Run

1. **Install Dependencies:**  Ensure you have Python 3 installed. Then, install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Main Script:** Execute the main script using:

   ```bash
   python main.py
   ```

## Example Output

The script will print a summary of the analysis to the console, including key findings and model performance metrics.  Additionally, the script generates several visualizations, including:

* **Sentiment Analysis Summary:** A bar chart visualizing the overall sentiment distribution across all reviews.
* **Feature Importance Plot:** A bar chart showing the relative importance of different features in predicting restaurant success.
* **Model Performance Metrics:** Printed to the console, including accuracy, precision, recall, and F1-score.

These visualizations are saved as PNG files in the `output` directory.  The exact filenames may vary depending on the specific analysis.  For example, `sentiment_distribution.png` and `feature_importance.png` are likely to be generated.