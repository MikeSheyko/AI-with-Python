import json
import matplotlib.pyplot as plt
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Завантаження даних
df = pd.read_csv(
    './assets/product-reviews-list_product-reviews_captured-list_2026-10-24_10-12-31_01a16cc0-15e6-7b67-a975-944043fafb56.csv'
)

# Видаляємо рядки без тексту відгуку
df = df.dropna(subset=['Review Text']).copy()
df['Review Text'] = df['Review Text'].astype(str)

# NLP
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Токенізація та видалення стоп-слів
def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return tokens

# Стемінг
def stem_text(tokens):
    return [stemmer.stem(word) for word in tokens]

# Лематизація
def lemmatize_text(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

# Обробка відгуків
df['Tokens'] = df['Review Text'].apply(tokenize_text)
df['Stemmed'] = df['Tokens'].apply(stem_text)
df['Lemmatized'] = df['Tokens'].apply(lemmatize_text)

df['Tokens'] = df['Tokens'].apply(lambda words: ' '.join(words))
df['Stemmed'] = df['Stemmed'].apply(lambda words: ' '.join(words))
df['Lemmatized'] = df['Lemmatized'].apply(lambda words: ' '.join(words))

# Аналіз тональності
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        sentiment = 'Positive'
    elif polarity < -0.1:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return polarity, sentiment

df[['Polarity', 'Sentiment']] = df['Review Text'].apply(
    lambda text: pd.Series(analyze_sentiment(text))
)

# Підрахунок кількості відгуків за категоріями
sentiment_counts = df['Sentiment'].value_counts()

summary = {
    'Total reviews': len(df),
    'Positive': int(sentiment_counts.get('Positive', 0)),
    'Neutral': int(sentiment_counts.get('Neutral', 0)),
    'Negative': int(sentiment_counts.get('Negative', 0))
}

# Збереження результатів у CSV
df.to_csv(
    './assets/processed_reviews.csv',
    index=False,
    encoding='utf-8'
)

# Збереження результатів у JSON
with open('./assets/sentiment_summary.json', 'w', encoding='utf-8') as file:
    json.dump(summary, file, indent=4, ensure_ascii=False)

# Виведення результатів
print('Sentiment analysis results:')
print(f"Total reviews: {summary['Total reviews']}")
print(f"Positive reviews: {summary['Positive']}")
print(f"Neutral reviews: {summary['Neutral']}")
print(f"Negative reviews: {summary['Negative']}")

print('\nFirst five processed reviews:')
print(df[['Review Text', 'Tokens', 'Stemmed',
          'Lemmatized', 'Polarity', 'Sentiment']].head())

print('\nFiles saved successfully!')

# Графік
sentiments = ['Positive', 'Neutral', 'Negative']
counts = [
    summary['Positive'],
    summary['Neutral'],
    summary['Negative']
]

plt.bar(sentiments, counts)
plt.title('Sentiment Analysis of Product Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')

for i, count in enumerate(counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()