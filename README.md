# NLP_TwitterSentimentAnalysis

## Data Sources


## Overview
This repository focuses on conducting sentiment analysis on user-generated content related to Android and Apple products. Dive into the world of consumer opinions, feedback, and reviews to gain valuable insights into the perception and sentiment surrounding these two tech giants using Machine learning models.

## Table of Contents
- Description
- Features
- Installation
- Data Sources
- Data Processing
- Model Training
- Evaluation
- Results
- Technologies Used
- License

## Description
Our objectives encompass a range of brand-related challenges, all with the common goal of leveraging Twitter data to gain valuable insights. Our diverse audience, including brand managers, marketing strategists, product launch teams, customer support, influencer marketing, and seasonal marketing teams, will collectively benefit from these analyses. We aim to enhance our brand's competitive edge by comparing sentiment and customer perceptions against competitors, evaluate the success of recent product launches, improve customer support through feedback analysis, identify potential brand influencers, and tailor seasonal marketing campaigns to customer sentiment. By addressing these multifaceted aspects of brand management, we seek to maximize our brand's impact and resonate more effectively with our target audience.

## Features
- Data processing
- Text preprocessing
- Data visualization
- Model training
- Evaluation

## Installation
### Libraries
```bash
pip install pandas
pip install matplotlib
pip install seaborn
pip install nltk
pip install wordcloud
pip install scikit-learn
pip install tensorflow
pip install torch
pip install transformers
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


## Data Processing & Text Preprocessing
In data processing, we prepare and clean our dataset for analysis. This involves tasks like handling missing data, removing duplicates, and transforming data into a usable format. It ensures that the data is accurate and ready for analysis. Text preprocessing is specifically applied to text data. It involved tasks like tokenization (breaking text into words or phrases), removing stopwords (common words like "the" or "and"), and stemming (reducing words to their root form). For my case, I used Lemmatization over stemming. Text preprocessing makes text data suitable for natural language processing (NLP) tasks like sentiment analysis or topic modeling.

## Model Training
I conducted a baseline logistic regression analysis to categorize sentiment in text into three classes (1.0, 2.0, and 3.0). The precision for class 1.0 is 62%, while classes 2.0 and 3.0 have lower precision. Class 3.0 exhibits the highest recall at 88%, whereas class 2.0 has a lower recall. The F1-scores reflect a balance between precision and recall, with class 3.0 achieving the highest score (78%). The model's overall accuracy in sentiment classification is 68%.

In the Random Forest Classifier, key hyperparameters were fine-tuned. 'C' (regularization parameter) was set to 1, signifying low regularization. The 'kernel' used was 'rbf' for capturing nonlinear data patterns.

The SVM model achieved an accuracy of 0.69, correctly predicting 69% of instances. Class 1 showed decent precision but lower recall, indicating challenges in identifying this class. Class 3 performed well with high precision and recall. However, Class 2 had lower performance overall. In summary, the model excels in predicting Class 3 but struggles with Class 2.

For the final model, I chose an RNN model where:
Model Type: The Sequential Model employs three key layers: the Embedding Layer, which transforms words into 100-dimensional numerical vectors; the SimpleRNN Layer, which deciphers sequential data patterns into 128-dimensional representations; and the Dense Layer, facilitating decision-making among three categories. In summary, this Sequential Model interprets words, identifies text patterns, and classifies text into categories (e.g., positive, negative, or neutral sentiment) based on extensive training data.

## Evaluation & Results

## Technologies Used

### Data Collection and Preprocessing
- **Pandas**: Pandas is used for data manipulation and analysis. We utilize it to read, clean, and preprocess the Twitter data.

### Data Visualization
- **Matplotlib**: Matplotlib is employed for creating various data visualizations, including plots and charts that help us understand data patterns and trends.
- **Seaborn**: Seaborn complements Matplotlib by providing an aesthetically pleasing and high-level interface for creating informative and attractive statistical graphics.

### Natural Language Processing (NLP)
- **NLTK (Natural Language Toolkit)**: NLTK is used for various NLP tasks such as tokenization, removing stopwords, and stemming. It helps us prepare text data for sentiment analysis.
- **WordCloud**: WordCloud is used to create word clouds, which visually represent the most frequent words in the text data.
- **Scikit-Learn**: Scikit-Learn provides machine learning tools, including algorithms for sentiment analysis and model evaluation.
- **TensorFlow and Torch**: These deep learning frameworks are used for building and training neural network models, such as RNN and BERT, for advanced sentiment analysis.
- **Transformers**: Transformers are specifically used for working with pre-trained models like BERT for text classification.
- **PyTorch:**. It is a deep learning framework, used for creating and training neural networks.
### Methodology
1. **Data Collection**: We gathered user-generated content related to Android and Apple products from Twitter.
2. **Data Preprocessing**: Pandas, NLTK, and other libraries were used to clean and prepare the data. Text data was tokenized, stopwords were removed, and text was transformed into numerical representations.
3. **Data Visualization**: Matplotlib and Seaborn helped us visualize data trends and patterns.
4. **Sentiment Analysis**: Scikit-Learn was used for baseline models like Logistic Regression and Random Forest. TensorFlow and Torch were used for advanced models like RNN and BERT.
5. **Model Evaluation**: We assessed model performance using accuracy, precision, recall, F1-score, and confusion matrices.
6. **Insights Generation**: The results of our analysis provide valuable insights into consumer sentiment and perceptions of Android and Apple products.

By employing these technologies and methodologies, we were able to conduct comprehensive sentiment analysis on Twitter data and extract meaningful insights for brand management and marketing strategies.

## Usage
Our project serves a wide range of brand-related objectives, catering to diverse audiences across the organization.
Brand managers and marketing strategists can harness the power of our analysis to gain a competitive edge by comparing sentiments and customer perceptions against competitors, refining strategies, and uncovering strengths and weaknesses.

Product launch teams, marketing departments, and product managers benefit from our insights into customer sentiment during recent product launches, enabling them to gauge the effectiveness of strategies and identify areas for enhancement.

Customer support teams, customer experience managers, and quality assurance departments can analyze feedback from negative sentiments to enhance customer support services and address recurring pain points.

Influencer marketing teams and social media managers can identify potential brand influencers among those expressing strong positive sentiments, fostering brand advocacy.

Seasonal marketing teams, digital marketing managers, and brand strategists can utilize sentiment-based insights to tailor marketing campaigns for specific seasons and holidays, aligning with customer preferences.

Incorporating these multifaceted aspects of brand management, our project aims to empower teams across the organization to maximize our brand's impact and connect more effectively with our target audience.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE.md] file for details.
## Acknowledgments
