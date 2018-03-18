import json
import keras
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adadelta
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.models import load_model

Xtokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, )
Ytokenizer = Tokenizer(filters=' ', lower=True, )
model = Sequential()
file_path = 'engines/datasets/tmdb_5000_movies.csv'


def extractGenres(row):
    json_dict_list = json.loads(row.genres)
    genres = [d['name'].lower() for d in json_dict_list if 'name' in d]
    return " ".join(genres)


df = pd.read_csv(file_path)
X = [str(p.overview) for p in df.itertuples()]
y = [extractGenres(p).rstrip() for p in df.itertuples()]
Xtokenizer.fit_on_texts(X)
Ytokenizer.fit_on_texts(y)
plot_matrix = Xtokenizer.texts_to_matrix(X)
genre_matrix = Ytokenizer.texts_to_matrix(y)
input_size = plot_matrix.shape[1]
output_size = genre_matrix.shape[1]
split_ratio = 0.9
split_index = int(len(X) * split_ratio)
X_train, X_test = np.array(plot_matrix[:split_index]), np.array(plot_matrix[split_index:])
y_train, y_test = np.array(genre_matrix[:split_index]), np.array(genre_matrix[split_index:])
model.add(Dense(512, activation = 'relu', input_dim = input_size))
model.add(Dropout(0.4))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(output_size, activation = 'sigmoid'))
model.summary()
#adadelta = Adadelta(lr =1.0, rho = 0.95, decay = 0.0)
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics = ['top_k_categorical_accuracy'])
checkpoint = ModelCheckpoint(filepath=None, monitor='val_loss', mode='min', verbose=1)
model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=0, callbacks=[])
score = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
model.save('engines/el_model.h5')
print("Accuracy: ", score)


def getMovies(text):
    print("Model chal raha hai"+text)	
    text_test = Xtokenizer.texts_to_matrix([text])
    text_probs = model.predict(np.array(text_test))
    print(text_probs)	
    text_genres = [indices_to_genres(probs_to_indices(p, top_k)) for p in text_probs]
    return genres_to_top_k_movies(text_genres[0])


top_k = 3
probs_to_indices = lambda ypb, k: np.argpartition(ypb, -k)[-k:]
genre_to_index = Ytokenizer.word_index
index_to_genre = [0] * output_size
genre_to_index[''] = 0
for k, v in genre_to_index.items():
    index_to_genre[v] = k
indices_to_genres = lambda ind: [index_to_genre[i] for i in ind]

dframe = pd.read_csv(file_path)
genre_to_movies_dict = {}
movies_popularity_dict = {}  # storing movies with their corresponding calculated scores.
index_to_movie_list = []
# Parameters for calculating weighted review:
C = 6.9  # mean vote across whole report
M = 3000  # minimum votes required to be listed in Top 250 by IMDB.
for row in dframe.itertuples():
    V = row.vote_count
    R = row.vote_average
    WR = (V / (V + M)) * R + (M / (V + M)) * C
    movies_popularity_dict[row.title] = WR
    index_to_movie_list.append(row.title)
    genre_map = json.loads(row.genres)
    genres = [d['name'].lower() for d in genre_map if 'name' in d]
    for genre in genres:
        if not genre in genre_to_movies_dict:
            genre_to_movies_dict[genre] = []
        genre_to_movies_dict[genre].append(row.title)
movie_to_index = {}
for i, movie in enumerate(movies_popularity_dict):
    movie_to_index[movie] = i
count_to_indices = lambda counter, k: np.argpartition(counter, -k)[-k:]
print(movie_to_index)

def genres_to_top_k_movies(genres, k=5):
    print(genres)
    counter = [0] * len(movies_popularity_dict)
    for genre in genres:
        for movie in genre_to_movies_dict[genre]:
            counter[movie_to_index[movie]] += 1
    count_pop_tuple_list = []
    for i, count in enumerate(counter):
        tuple = (count, movies_popularity_dict[index_to_movie_list[i]], index_to_movie_list[i])
        count_pop_tuple_list.append(tuple)
    final_sorted_tuple_list = sorted(count_pop_tuple_list, key=lambda tuple: (tuple[0], tuple[1]))
    output_movies = []
    iter = 0		
    for tuple in reversed(final_sorted_tuple_list):
        iter += 1
        if iter <= 5:
            output_movies.append(tuple[2])
        else:
            break
    return output_movies
