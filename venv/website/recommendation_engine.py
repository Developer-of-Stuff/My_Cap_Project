# get_ipython().system('pip install numpy')
# get_ipython().system('pip install pandas')
# get_ipython().system('pip install sklearn')
# get_ipython().system('pip install torch')
# get_ipython().system('pip install torchvision')
# get_ipython().system('pip install fuzzywuzzy')
# get_ipython().system('pip install python-Levenshtein')

import dill
import pandas as pd
import wget
from data_loading import Loader, MatrixFactorization
from torch import no_grad
from torch.cuda import is_available
from sklearn.preprocessing import StandardScaler
from fuzzywuzzy.process import extractOne


class RecommendationEngine:
    def __init__(self, games_url, ratings_url, test_data_url):
        games_file = wget.download(games_url)
        ratings_file = wget.download(ratings_url)
        test_data_raw = wget.download(test_data_url)

        self.steam_games = pd.read_csv(games_file)
        self.user_ratings = pd.read_csv(ratings_file)

        with open(test_data_raw, 'rb') as raw_data:
            test_data = dill.load(raw_data)

        self.similarities = test_data['similarities']
        self.complete_set = dl.Loader(self.user_ratings)
        self.genre_avg_hrs = test_data['genre_avg_hrs']
        self.dev_ratings = test_data['dev_ratings']
        self.model = test_data['model']

    def recommender(self, name_input, n_games_requested):
        output = []
        output_steam_indices = []
        cuda = is_available()
        ur_index = extractOne(name_input, self.user_ratings['game_title'])[2]
        appid = self.user_ratings.iloc[ur_index].appid
        steam_index = self.steam_games.loc[self.steam_games.appid == appid].index.values[0]

        similarity_scores = {}
        for i in range(len(self.similarities[steam_index])):
            similarity_scores[self.steam_games.iloc[i].appid] = self.similarities[steam_index][i]

        steam_name = self.steam_games.loc[self.steam_games.appid == appid]['name'].values[0]

        loader_index = self.complete_set.appid_to_index[appid]
        user_data = self.complete_set.get_user_data(loader_index)
        with no_grad():
            self.model.eval()
            score_to_idx = {}
            n_users = len(user_data[0])
            recom_games_indices = []
            for game_data in user_data:
                if cuda:
                    user_pred = self.model(game_data.cuda()).cpu().numpy()
                else:
                    user_pred = self.model(game_data).numpy()

                transformer = StandardScaler()
                transformer.fit(user_pred.reshape(-1, 1))
                transformed_values = transformer.transform(user_pred.reshape(-1, 1))

                average = 0
                for pred in transformed_values:
                    average += pred[0]
                average /= n_users

                game_sim_score = similarity_scores[self.complete_set.index_to_appid[game_data.numpy()[0][1]]]
                score_to_idx[average + game_sim_score / 10] = game_data.numpy()[0][1]
            for i in range(n_games_requested):
                max_idx = score_to_idx[max(score_to_idx)]
                recom_games_indices.append(max_idx)
                score_to_idx.pop(max(score_to_idx))

            for idx in recom_games_indices:
                game_appid = self.complete_set.index_to_appid[idx]
                output.append(self.steam_games.loc[self.steam_games.appid == game_appid].name.values[0])
                output_steam_indices.append(self.steam_games.loc[self.steam_games.appid == game_appid].index.values[0])
        return [steam_index, appid, steam_name, output, output_steam_indices]

    def get_model(self):
        return self.model

    def get_cos_sim(self):
        return self.similarities

    def get_genre_avg_hrs(self):
        return self.genre_avg_hrs

    def get_dev_ratings(self):
        return self.dev_ratings

    def get_steam_games(self):
        return self.steam_games

    def get_user_ratings(self):
        return self.user_ratings




