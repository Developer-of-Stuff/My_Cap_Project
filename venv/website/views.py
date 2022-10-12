import datatools
from flask import Blueprint, render_template, request

views = Blueprint('views', __name__)
engine = datatools.RecommendationEngine(
    'https://heroku-s3-capstone-bucket.s3.us-east-2.amazonaws.com/edited_steam_games.csv',
    'https://heroku-s3-capstone-bucket.s3.us-east-2.amazonaws.com/indexed_ratings_appid.csv',
    'https://heroku-s3-capstone-bucket.s3.us-east-2.amazonaws.com/train_test_data.pkl'
)


game_title = None
model_output = None
steam_index = None
steam_name = None
cos_values = None


@views.route('/', methods=['GET', 'POST'])
def home():
    global game_title, steam_name, steam_index, model_output, cos_values
    ratings_list = engine.get_ratings_list()

    if request.method == 'POST':
        user_input = request.form.get('user-input')
        if game_title is None and user_input != "":
            game_title = user_input
            steam_index, appid, steam_name, model_output, cos_values = engine.recommender(game_title, 5)
        elif game_title != user_input and user_input != "":
            game_title = user_input
            steam_index, appid, steam_name, model_output, cos_values = engine.recommender(game_title, 5)

        if model_output is not None:
            new_cos_values = []
            for value in cos_values:
                value *= 10000
                value = round(value, 0)
                value /= 100
                new_cos_values.append(value)
            return render_template(
                "base.html",
                steam_name=steam_name,
                output=model_output['title'],
                cos_values=new_cos_values,
                hrs_v_rating=ratings_list,
                game_hours=[model_output['title'], model_output['avg_hours']],
                rating_per_rec=[model_output['title'], model_output['rating']],
                game_indices=model_output['steam_index']
            )
    return render_template("base.html")
