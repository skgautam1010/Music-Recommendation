from operator import ne
from flask.helpers import url_for
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.sparse import csr_matrix
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from flask import Flask, render_template, request, flash
from werkzeug.utils import redirect
from flask_sqlalchemy import SQLAlchemy


df_songs = pd.read_csv('songs.csv')
df_songs.head()


df_songs.shape
#print(f'There are {df_songs.shape[0]} observations in the dataset')

df_songs.isnull().sum()
df_songs.dtypes
unique_songs = df_songs['title'].unique().shape[0]
unique_songs
unique_artist = df_songs['artist_name'].unique().shape[0]
unique_artist
unique_users = df_songs['user_id'].unique().shape[0]
unique_users
ten_pop_songs = df_songs.groupby('title')['listen_count'].count(
).reset_index().sort_values(['listen_count', 'title'], ascending=[0, 1])
ten_pop_songs['percentage'] = round(ten_pop_songs['listen_count'].div(
    ten_pop_songs['listen_count'].sum())*100, 2)
ten_pop_songs = ten_pop_songs[:10]
ten_pop_songs
labels = ten_pop_songs['title'].tolist()
counts = ten_pop_songs['listen_count'].tolist()
# plt.figure(figsize=(10,7))
# sns.barplot(x=counts,y=labels,palette='bright')
# sns.despine(left=True,bottom=True)
# plt.bar(labels,counts,color='maroon',width=0.4)
# artist 10 most popular artist
ten_pop_artist = df_songs.groupby('artist_name')['listen_count'].count(
).reset_index().sort_values(['listen_count', 'artist_name'], ascending=[0, 1])
ten_pop_artist = ten_pop_artist[:10]
ten_pop_artist
counts = ten_pop_artist['listen_count'].tolist()
labels = ten_pop_artist['artist_name'].tolist()

# plt.figure(figsize=(15,9))
# sns.barplot(x=counts,y=labels,palette='dark')
# sns.despine(left=True,bottom=True)
listen_counts = pd.DataFrame(df_songs.groupby(
    'listen_count').size(), columns=['count'])
listen_counts.reset_index(drop=False)['listen_count'].iloc[-1]
df_songs['listen_count'].mean()
#print(f"on an average a user listens to {round(df_songs['listen_count'].mean())} times")
# plt.figure(figsize=(20,6))
# plt.boxplot(df_songs['listen_count'])
# sns.boxplot(x='listen_count',data=df_songs)
# sns.despine()
song_user = df_songs.groupby('user_id')['song_id'].count()
#print(f"A user listens to an average of {np.mean(song_user)} songs")
#print(f"{np.median(song_user)} songs, with minimum {np.min(song_user)} and maximum {np.max(song_user)} songs")
values_matrix = unique_users * unique_songs
values_matrix
df_songs.shape[0]
zero_values_matrix = values_matrix - df_songs.shape[0]
zero_values_matrix
#print(f"The matrix of users x songs has {zero_values_matrix} values that are zero")
song_ten_id = song_user[song_user > 16].index.to_list()
df_song_id_more_ten = df_songs[df_songs['user_id'].isin(
    song_ten_id)].reset_index(drop=True)
df_song_id_more_ten.head()
df_songs_features = df_song_id_more_ten.pivot(
    index='song_id', columns='user_id', values='listen_count').fillna(0)

# obtain a sparse matrix
mat_songs_features = csr_matrix(df_songs_features.values)
df_songs_features.head()
df_unique_songs = df_songs.drop_duplicates(
    subset=['song_id']).reset_index(drop=True)[['song_id', 'title']]
decode_id_song = {
    song: i for i, song in
    enumerate(list(df_unique_songs.set_index(
        'song_id').loc[df_songs_features.index].title))
}


class Recommender:
    def __init__(self, metric, algorithm, k, data, decode_id_song):
        self.metric = metric
        self.algorithm = algorithm
        self.k = k
        self.data = data
        self.decode_id_song = decode_id_song
        self.model = self._recommender().fit(data)

    def make_recommendation(self, new_song, n_recommendations):
        recommended = self._recommend(
            new_song=new_song, n_recommendations=n_recommendations)
        #print("... Done")
        return recommended

    def _recommender(self):
        return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1)

    def _recommend(self, new_song, n_recommendations):
        # Get the id of the recommended songs
        recommendations = []
        recommendation_ids = self._get_recommendations(
            new_song=new_song, n_recommendations=n_recommendations)
        # return the name of the song using a mapping dictionary
        recommendations_map = self._map_indeces_to_song_title(
            recommendation_ids)
        # Translate this recommendations into the ranking of song titles recommended
        for i, (idx, dist) in enumerate(recommendation_ids):
            recommendations.append(recommendations_map[idx])
        return recommendations

    def _get_recommendations(self, new_song, n_recommendations):
        # Get the id of the song according to the text
        recom_song_id = self._fuzzy_matching(song=new_song)
        # Start the recommendation process
        #print(f"Starting the recommendation process for {new_song} ...")
        # Return the n neighbors for the song id
        distances, indices = self.model.kneighbors(
            self.data[recom_song_id], n_neighbors=n_recommendations+1)
        return sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

    def _map_indeces_to_song_title(self, recommendation_ids):
        # get reverse mapper
        return {song_id: song_title for song_title, song_id in self.decode_id_song.items()}

    def _fuzzy_matching(self, song):
        match_tuple = []
        # get match
        for title, idx in self.decode_id_song.items():
            ratio = fuzz.ratio(title.lower(), song.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print(
                f"The recommendation system could not find a match for {song}")
            return
        return match_tuple[0][1]


app = Flask(__name__)


app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/flask_learning'
db = SQLAlchemy(app)


class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(50), unique=False, nullable=False)
    lname = db.Column(db.String(50), unique=False, nullable=False)
    contact = db.Column(db.String(12), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    msg = db.Column(db.String(100), unique=False, nullable=False)


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")


@app.route("/contactus", methods=['GET', 'POST'])
def contactus():
    if request.method == 'POST':
        f_name = request.form.get('firstname')
        l_name = request.form.get('lastname')
        contactus = request.form.get('telnum')
        emailid = request.form.get('emailid')
        feedback = request.form.get('feedback')

        entry = Contact(fname=f_name, lname=l_name,
                        contact=contactus, email=emailid, msg=feedback)
        db.session.add(entry)
        db.session.commit()
    return render_template("contactus.html")


@app.route('/', methods=["POST", "GET"])
def display():
    if request.method == "POST":
        song_name = request.form["sng"]
        song = song_name
        model = Recommender(metric='cosine', algorithm='brute', k=20,
                            data=mat_songs_features, decode_id_song=decode_id_song)
        new_recommendations = model.make_recommendation(
            new_song=song, n_recommendations=9)
    if(new_recommendations==0):
        return render_template('index.html')
    else:
        return render_template('index.html', new_recommendations=new_recommendations,song=song)


app.run(debug=True)
