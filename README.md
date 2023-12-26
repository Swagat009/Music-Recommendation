# Flask Music Recommendation System

This project implements a music recommendation system using Flask, pandas, and scikit-learn. The recommendation system is based on cosine similarity, and the web application allows users to input a song title to receive personalized music recommendations.

## Dependencies

- pandas: Data manipulation library
- Flask: Web framework for building web applications
- scikit-learn: Machine learning library for data preprocessing and cosine similarity calculation

## Dataset

Two datasets are used in this project:

1. `dataset.csv`: Main dataset containing information about music tracks.
2. `data/data.csv`: Additional dataset with track ID and corresponding release year.

## Data Preprocessing

The datasets are loaded and merged on the 'track_id' column. Duplicate values based on 'track_id' are checked and removed. The genre information is encoded using one-hot encoding and concatenated with the original dataframe. Numerical features are scaled using MinMaxScaler.

## Cosine Similarity Calculation

Cosine similarity is calculated between music tracks using features such as explicitness, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and release year.

## Recommendation Function

The `get_recommendations` function takes a song title as input and generates music recommendations based on cosine similarity scores. The recommendations include the song name, artists, album name, and similarity score.

## Flask Web Application

The Flask application consists of two routes:

1. `/`: Renders the home page with an HTML form to input a song title.
2. `/recommend`: Handles POST requests with a song title, calls the recommendation function, and renders a page with the recommended songs.

## How to Run

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the Flask application using `python app.py`.
3. Access the application in a web browser at `http://127.0.0.1:5000/`.

## Usage

1. Enter a song title in the input form on the home page.
2. Click the "Recommend" button.
3. View the recommended songs on the results page, including song name, artists, album name, and similarity score.

Feel free to explore, modify, and contribute to enhance the music recommendation system!

---

**Note:** Make sure to replace the dataset filenames and adjust the file paths accordingly in the code.

