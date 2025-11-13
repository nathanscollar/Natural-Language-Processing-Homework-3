# preprocess.py

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt_tab')

def load_and_preprocess(file_path):
    # read data from file path
    print("Loading dataset from:", file_path)
    imdb_df = pd.read_csv(file_path)
    print("Original shape:", imdb_df.shape)

    # preprocess text by converting to lowercase and removing punctuation
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    # preprocess all reviews
    imdb_df['review'] = imdb_df['review'].apply(preprocess_text)

    # split into training and testing
    train_df = imdb_df.iloc[:25000].copy()
    test_df = imdb_df.iloc[25000:].copy()

    # tokenization
    train_df['tokens'] = train_df['review'].apply(word_tokenize)
    test_df['tokens'] = test_df['review'].apply(word_tokenize)

    # Use Keras Tokenizer to keep 10000 most frequent words
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df['tokens'])

    # conversion of text to sequence IDs
    train_seq = tokenizer.texts_to_sequences(train_df['tokens'])
    test_seq = tokenizer.texts_to_sequences(test_df['tokens'])

    # pad sequences
    def pad_all(train_seq, test_seq):
        return {
            25: (
                pad_sequences(train_seq, maxlen=25, padding='post', truncating='post'),
                pad_sequences(test_seq, maxlen=25, padding='post', truncating='post')
            ),
            50: (
                pad_sequences(train_seq, maxlen=50, padding='post', truncating='post'),
                pad_sequences(test_seq, maxlen=50, padding='post', truncating='post')
            ),
            100: (
                pad_sequences(train_seq, maxlen=100, padding='post', truncating='post'),
                pad_sequences(test_seq, maxlen=100, padding='post', truncating='post')
            ),
        }

    padded_data = pad_all(train_seq, test_seq)

    # encode output labels as 0 and 1
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df['sentiment'])
    y_test = encoder.transform(test_df['sentiment'])

    return padded_data, y_train, y_test