import os

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras import utils
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras.callbacks import TensorBoard
from time import time
from utils import generate_tweets_csv, convert_text_to_indices


def build_model(max_words):
    """
    Generating model using keras sequential layer
    :param max_words:
    :return:
    """
    K.clear_session()
    print('Generating Model...')
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(max_words,)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model


if __name__ == "__main__":
    print('Reading and generating csv files from xml file...')
    input_xml_path = os.path.join('dataset', 'en')
    input_label_path = os.path.join('dataset', 'en', 'truth.txt')

    dict_path = os.path.join('output', 'en_dictionary.json')
    tweet_path = os.path.join('output', 'twitter_en.csv')

    words_to_retain = 3000

    # Generate CSV file
    generate_tweets_csv(input_xml_path, input_label_path, tweet_path)

    # Read the CSV
    my_df = pd.read_csv(tweet_path)
    print(f'Unique personalities are {my_df.gender.unique()}')

    # label encode
    my_df['gender'] = my_df['gender'].map({'female': 0, 'male': 1, 'bot': 2})

    # Remove stop words
    stop_words = stopwords.words('english')
    my_df['text'] = my_df['text'].apply(lambda x: " ".join(x for x in x.split(" ") if x not in stop_words))

    print('Removing frequently used words')
    # Remove frequently occurring words
    freq = pd.Series(' '.join(my_df['text']).split()).value_counts()[:50]
    my_df['text'] = my_df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    print('Tokenizing the data')
    # Tokenize the data
    train_x = my_df.text
    train_y = my_df.gender
    tokenizer = Tokenizer(num_words=words_to_retain)
    tokenizer.fit_on_texts(train_x)
    dictionary = tokenizer.word_index

    print('Converting to word indices')
    # Convert text to word indices
    allWordIndices = []
    for text in tqdm(train_x):
        wordIndices = convert_text_to_indices(text, dictionary)
        allWordIndices.append(wordIndices)
    allWordIndices = np.asarray(allWordIndices)

    # One hot encode the data
    train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')

    # Convert labels to categories
    train_y = utils.to_categorical(train_y, 3)

    model = build_model(words_to_retain)

    # Adding callbacks
    tensorboard = TensorBoard(log_dir='logs\{}'.format(time()))
    model.fit(train_x, train_y,
              batch_size=32,
              epochs=2,
              verbose=1,
              validation_split=0.2,
              shuffle=True, callbacks=[tensorboard])

    model_save_name = 'my_model_05.hdf5'
    print(f'Model is saved in root folder with name {model_save_name}')
    model.save(model_save_name)

    print('To visualize the model: Please run the following command in a terminal in root folder of the project')
    print('tensorboard --logdir=logs/')
    # To visualize the model ->
    # Run `tensorboard --logdir=logs/`








