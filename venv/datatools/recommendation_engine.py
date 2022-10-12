
import dill
import pandas as pd
import wget
from datatools.data_loading import Loader
from torch import no_grad
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import mean, std
from fuzzywuzzy.process import extractOne


class RecommendationEngine:
    def __init__(self, games_url, ratings_url, test_data_url):
        try:
            self.steam_games = pd.read_csv('edited_steam_games.csv')
            self.user_ratings = pd.read_csv('indexed_ratings_appid.csv')
            with open('train_test_data.pkl', 'rb') as raw_data:
                test_data = dill.load(raw_data)
            print("Files located")
        except FileNotFoundError:
            print("Proceeding with file downloads:")
            games_file = wget.download(games_url)
            print("- edited_steam_games.csv downloaded")
            ratings_file = wget.download(ratings_url)
            print("- indexed_ratings_appid.csv downloaded")
            test_data_raw = wget.download(test_data_url)
            print("- train_test_data.pkl downloaded")
            self.steam_games = pd.read_csv(games_file)
            self.user_ratings = pd.read_csv(ratings_file)
            with open(test_data_raw, 'rb') as raw_data:
                test_data = dill.load(raw_data)

        features = []
        for i in range(self.steam_games.shape[0]):
            features.append(self.steam_games['name'][i] + ' ' + self.steam_games['developers'][i] + ' ' + self.steam_games['tags'][i])

        self.steam_games['combined_features'] = features
        self.count_matrix = CountVectorizer().fit_transform(self.steam_games['combined_features'])

        self.complete_set = Loader(self.user_ratings)
        self.model = test_data['model']
        self.game_hours = test_data['hours_by_game']
        self.ratings_list = {}

        for appid in self.game_hours:
            if appid in self.steam_games['appid'].values:
                if self.steam_games.loc[self.steam_games.appid == appid].rating.values[0] is not None:
                    self.ratings_list[self.game_hours[appid][1] / self.game_hours[appid][0]] = self.steam_games.loc[self.steam_games.appid == appid].rating.values[0]

        rat_list_keys = list(self.ratings_list.keys())
        ratings_mean, ratings_std = mean(rat_list_keys), std(rat_list_keys)
        cut_off = ratings_std * 3
        lower, upper = ratings_mean - cut_off, ratings_mean + cut_off
        outliers = [x for x in rat_list_keys if x < lower or x > upper]

        for value in outliers:
            for point in self.ratings_list:
                if point == value:
                    self.ratings_list.pop(point)
                    break

    def recommender(self, name_input, n_games_requested):
        output = {'steam_index': [], 'title': [], 'appid': [], 'rating': [], 'avg_hours': []}
        ur_index = extractOne(name_input, self.user_ratings['game_title'])[2]
        appid = self.user_ratings.iloc[ur_index].appid
        steam_index = self.steam_games.loc[self.steam_games.appid == appid].index.values[0]

        orig_similarities = cosine_similarity(self.count_matrix)[steam_index]
        similarities = list(enumerate(orig_similarities))
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[1:51]

        game_list_check = []
        for value in similarities:
            check_idx = value[0]
            check_appid = self.steam_games.iloc[check_idx].appid
            game_list_check.append(check_appid)

        steam_name = self.steam_games.loc[self.steam_games.appid == appid]['name'].values[0]
        loader_index = self.complete_set.appid_to_index[appid]
        user_data = self.complete_set.get_user_data(loader_index, game_list_check)
        with no_grad():
            self.model.eval()
            score_to_idx = {}
            loader_idx_avg = {}
            n_users = len(user_data[0])
            recom_games_indices = []
            edited_user_data = []

            for game_data in user_data:
                data_loader_idx = game_data.numpy()[0][1]
                data_appid = self.complete_set.index_to_appid[data_loader_idx]
                if data_appid in game_list_check:
                    edited_user_data.append(game_data)

            for game_data in edited_user_data:
                user_pred = self.model(game_data).numpy()
                transformed_values = StandardScaler().fit_transform(user_pred.reshape(-1, 1))

                average = 0
                for pred in transformed_values:
                    average += pred[0]
                average /= n_users

                loader_idx_avg[game_data.numpy()[0][1]] = average

            for loader_idx in loader_idx_avg:
                game_appid = self.complete_set.index_to_appid[loader_idx]
                game_steam_index = self.steam_games.loc[self.steam_games.appid == game_appid].index.values[0]

                game_sim_score = 0
                for value in similarities:
                    if game_steam_index == value[0]:
                        game_sim_score = value[1]
                        break

                score_to_idx[loader_idx_avg[loader_idx] + game_sim_score / 12] = loader_idx

            for i in range(n_games_requested):
                max_idx = score_to_idx[max(score_to_idx)]
                recom_games_indices.append(max_idx)
                score_to_idx.pop(max(score_to_idx))

            for idx in recom_games_indices:
                game_appid = self.complete_set.index_to_appid[idx]
                game_steam_games_entry = self.steam_games.loc[self.steam_games.appid == game_appid]
                output['steam_index'].append(game_steam_games_entry.index.values[0])
                output['title'].append(game_steam_games_entry.name.values[0])
                output['appid'].append(game_appid)
                output['rating'].append(game_steam_games_entry.rating.values[0])
                output['avg_hours'].append(self.game_hours[game_appid][1] / self.game_hours[game_appid][0])
        return [steam_index, appid, steam_name, output, orig_similarities]

    def get_model(self):
        return self.model

    def get_cos_sim(self):
        return self.similarities

    def get_game_hours(self):
        return self.game_hours

    def get_ratings_list(self):
        return self.ratings_list

    def get_steam_games(self):
        return self.steam_games

    def get_user_ratings(self):
        return self.user_ratings



