from flask import Blueprint, render_template, request
from .recommendation_engine import RecommendationEngine

views = Blueprint('views', __name__)
engine = RecommendationEngine('https://heroku-s3-capstone-bucket.s3.us-east-2.amazonaws.com/edited_steam_games.csv',
                              'https://heroku-s3-capstone-bucket.s3.us-east-2.amazonaws.com/indexed_ratings_appid.csv',
                              'https://heroku-s3-capstone-bucket.s3.us-east-2.amazonaws.com/train_test_data.pkl')


game_title = None
sg_titles = None
steam_index = None
steam_name = None
game_indices = None


@views.route('/', methods=['GET', 'POST'])
def home():
    global game_title, sg_titles, steam_name, steam_index, game_indices
    genre_hours = engine.get_genre_avg_hrs()
    dev_rating = engine.get_dev_ratings()

    if request.method == 'POST':
        user_input = request.form.get('user-input')
        if game_title is None and user_input != "":
            game_title = user_input
            steam_index, appid, steam_name, sg_titles, game_indices = engine.recommender(game_title, 5)
        elif game_title != user_input and user_input != "":
            game_title = user_input
            steam_index, appid, steam_name, sg_titles, game_indices = engine.recommender(game_title, 5)

        if sg_titles is not None:
            cos_values = engine.get_cos_sim()[steam_index].tolist()
            new_cos_values = []
            for value in cos_values:
                value *= 10000
                value = round(value, 0)
                value /= 100
                new_cos_values.append(value)
            return render_template("base.html", steam_name=steam_name, output=sg_titles, cos_values=new_cos_values, genre_hours=genre_hours, dev_rating=dev_rating, game_indices=game_indices)
    return render_template("base.html", genre_hours=genre_hours, dev_rating=dev_rating)
