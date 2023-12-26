import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
#pandas data manipulation
#Flask is a web framework for building web applications.
#request is used to access data sent with HTTP requests
#render_template is used to render HTML templates
#MinMaxScaler is used for feature scaling.
#cosine_similarity is used to calculate cosine similarities between items in a dataset.


#Create a Flask application:
app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv("dataset.csv")
df.drop(columns='Unnamed: 0', inplace=True)
#It loads a dataset from a CSV file, removes an unnamed column ('Unnamed: 0').

#Load another dataset and select relevant columns:
dfYear = pd.read_csv("data/data.csv")
dfYear = dfYear[['id', 'year']]
dfYear['track_id'] = dfYear['id']
dfYear.drop(columns='id', inplace=True)
#This code loads another dataset, selects 'id' and 'year' columns, renames 'id' to 'track_id', and drops the 'id' column.

#Merge the two datasets on the 'track_id' column:
df = pd.merge(df, dfYear, on='track_id')

# Check for duplicate values based on the 'track_id':
duplicates = df[df.duplicated('track_id')]
print("Duplicate Count:", len(duplicates))

# Crosstab Genre and Song
xtab_song = pd.crosstab(df['track_id'], df['track_genre'])
xtab_song = xtab_song * 2

# Concatenate the encoded genre columns with the original dataframe
dfDistinct = df.drop_duplicates('track_id')
dfDistinct = dfDistinct.sort_values('track_id')
dfDistinct = dfDistinct.reset_index(drop=True)

xtab_song.reset_index(inplace=True)
data_encoded = pd.concat([dfDistinct, xtab_song], axis=1)

#It defines a list of numerical feature names and scales these features using MinMaxScaler.
numerical_features = ['explicit', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'year']
scaler = MinMaxScaler()
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])

# Calculate item similarities
calculatied_features = numerical_features + list(xtab_song.drop(columns='track_id').columns)
cosine_sim = cosine_similarity(data_encoded[calculatied_features], data_encoded[calculatied_features])
#This code creates a list of features used to calculate cosine similarities and then computes the cosine similarity matrix for the dataset.

# Recommendation function
#This function takes a song title as input and generates music recommendations based on cosine similarity.
def get_recommendations(title, N=5):
    indices = pd.Series(data_encoded.index, index=data_encoded['track_name']).drop_duplicates()

    try:
        idx = indices[title]
        try:
            len(idx)
            temp = 2
        except:
            temp = 1
    except KeyError:
        return "Song not found in the dataset."
    
    if temp == 2:
        idx = indices[title][0]
    else:
        idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    song_indices = [i[0] for i in sim_scores]
    recommended_songs = data_encoded[['track_name', 'artists', 'album_name']].iloc[song_indices]

    sim_scores_list = [i[1] for i in sim_scores]
    recommended_list = recommended_songs.to_dict(orient='records')
    for i, song in enumerate(recommended_list):
        song['similarity_score'] = sim_scores_list[i]
    
    return recommended_list

#This route returns the 'index.html' template when the root URL is accessed
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form.get('title')
    recommended_songs = get_recommendations(title, N=5)
    if isinstance(recommended_songs, str):
        return recommended_songs
    else:
        return render_template('recommendations.html', recommended_songs=recommended_songs)
#This route is for handling song recommendation requests. It expects a POST request with a song title, calls the get_recommendations function, and then renders the 'recommendations.html' template with the recommended songs.

#This block starts the Flask application when the script is executed directly, enabling debugging mode.
if __name__ == '__main__':
    app.run(debug=True)
