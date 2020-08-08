'''Advanced Topic in Controls: Largew Scale Convex Optimization. Programming Exercise 11. The exercise is solved according to functional programming principles. '''

__author__ = 'Manuel Galliker'
__license__ = 'GPL'

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import math


def complete_matrix(matrix_shape, known_values, known_value_indices):
    X = cp.Variable(matrix_shape)
    objective = cp.norm(X, "nuc")
    constraints = [X[known_value_indices] == known_values]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.SCS, verbose=True, use_indirect=True)
    print("Optimal value: ", problem.value)
    # print("X:\n", X.value)
    # print("X:\n", X.value.shape)
    return pd.DataFrame(X.value)


def trim_complete_dataset(n_ani, n_rat):
    rating_df = pd.read_csv('rating_upload_psy_scifi_new_index.csv')
    anime_df = pd.read_csv('anime_upload_psy_scifi_new_index.csv')
    rating = rating_df.to_numpy()
    rating[:, 0] = rating[:, 0]-1
    rating[:, 1] = rating[:, 1]-1
    rating = rating[rating[:, 1] <= n_ani-1, :]
    rating = rating[rating[:, 0] <= n_rat-1, :]
    anime_df = anime_df.iloc[range(500), :]
    anime_list = anime_df['name'].tolist()
    return rating, anime_df


def compute_known_values(rating):
    known_value_indices = []
    known_values = []
    for row in range(rating.shape[0]):
        current_indices = [rating[row, 0],
                           rating[row, 1]]
        known_value_indices.append(current_indices)
        known_values.append(rating[row, 2])
    known_value_indices = tuple(zip(*known_value_indices))
    return known_values, known_value_indices


def determine_top_recommendations(top_count, recommendation_frame, anime_df, already_watched_ids):
    recommendation_frame = recommendation_frame.drop(
        already_watched_ids, axis=0)
    sorted_frame = recommendation_frame.sort_values(ascending=False)
    top_anime_id = sorted_frame.iloc[range(top_count)]
    print(top_anime_id)
    ind_list = top_anime_id.index.values.tolist()
    top_recommendations = anime_df.iloc[ind_list, 1].reset_index()
    top_recommendations['index'] += 1
    top_recommendations.columns = ['anime id', 'name']
    return top_recommendations


def determine_recommendations_above_score(score, recommendation_frame, anime_df, already_watched_ids):
    recommendation_frame = recommendation_frame.drop(
        already_watched_ids, axis=0)
    sorted_frame = recommendation_frame.sort_values(ascending=False)
    top_anime_id = sorted_frame.loc[sorted_frame >= 6]
    print(top_anime_id)
    ind_list = top_anime_id.index.values.tolist()
    top_recommendations = anime_df.iloc[ind_list, 1].reset_index()
    top_recommendations['index'] += 1
    top_recommendations.columns = ['anime id', 'name']
    return top_recommendations


def trim_scifi_dataset(n_ani, n_rat):
    rating_df = pd.read_csv('rating_upload_scifi_new_index.csv')
    anime_df = pd.read_csv('anime_upload_scifi_new_index.csv')
    rating = rating_df.to_numpy()
    rating[:, 0] = rating[:, 0]-1
    rating[:, 1] = rating[:, 1]-1
    rating = rating[rating[:, 1] <= n_ani-1, :]
    rating = rating[rating[:, 0] <= n_rat-1, :]
    anime_df = anime_df.iloc[range(500), :]
    anime_list = anime_df['name'].tolist()
    return rating, anime_list


def main():

    # Task 1
    rating1, anime_df = trim_complete_dataset(500, 50)
    known_values, known_value_indices = compute_known_values(rating1)
    results1 = complete_matrix((50, 500), known_values, known_value_indices)
    print(results1)
    results1.to_csv('results1.csv')

    # Task 2
    mathias_anime_ratings2 = [[50, 1, 10], [
        50, 28, 5], [50, 136, 10], [50, 94, 6]]
    mathias_anime_ratings2_ids = [1, 28, 136, 94]
    rating2 = np.append(rating1, mathias_anime_ratings2, axis=0)
    known_values, known_value_indices = compute_known_values(rating2)
    results2 = complete_matrix((51, 500), known_values, known_value_indices)
    mathias_results = results2.loc[50, :]
    mathias_recom = determine_top_recommendations(
        5, mathias_results, anime_df, mathias_anime_ratings2_ids)
    mathias_recom.to_csv('results2.csv')
    print(mathias_recom)

    # Task 3
    mathias_anime_ratings3 = [[50, 1, 10], [50, 28, 5], [
        50, 136, 10], [50, 94, 6], [50, 30, 9]]
    mathias_anime_ratings3_ids = [1, 28, 136, 94, 30]
    rating3 = np.append(rating1, mathias_anime_ratings3, axis=0)
    known_values, known_value_indices = compute_known_values(rating3)
    results3 = complete_matrix((51, 500), known_values, known_value_indices)
    mathias_results = results3.loc[50, :]
    mathias_recom = determine_top_recommendations(
        5, mathias_results, anime_df, mathias_anime_ratings3_ids)
    mathias_recom.to_csv('results3.csv')
    print(mathias_recom)

    # Task 4
    rating4, anime_list4 = trim_scifi_dataset(500, 50)
    mathias_anime_ratings4 = [[50, 1, 10], [
        50, 22, 5], [50, 98, 10], [50, 69, 6]]
    mathias_anime_ratings4_ids = [1, 22, 98, 69]
    rating4 = np.append(rating1, mathias_anime_ratings4, axis=0)
    known_values, known_value_indices = compute_known_values(rating4)
    results3 = complete_matrix((51, 500), known_values, known_value_indices)
    mathias_results = results3.loc[50, :]
    mathias_recom = determine_top_recommendations(
        5, mathias_results, anime_df, mathias_anime_ratings4_ids)
    mathias_recom.to_csv('results4.csv')
    print(mathias_recom)
    mathias_recom_above_six = determine_recommendations_above_score(
        6, mathias_results, anime_df, mathias_anime_ratings4_ids)
    print(mathias_recom_above_six)
    mathias_recom_above_six.to_csv('Galliker_recommendations.txt', index=False)

    # Task 5
    mathias_anime_ratings5 = [[50, 1, 10], [
        50, 28, 5], [50, 136, 10], [50, 94, 6]]
    mathias_anime_ratings5_ids = [1, 28, 136, 94]
    rating5 = np.append(rating1, mathias_anime_ratings5, axis=0)
    known_values, known_value_indices = compute_known_values(rating5)
    for i in range(len(known_values)):
        if (known_values[i] >= 6):
            known_values[i] = 1
        else:
            known_values[i] = 0
    results2 = complete_matrix((51, 500), known_values, known_value_indices)
    mathias_results = results2.loc[50, :]
    mathias_recom = determine_top_recommendations(
        5, mathias_results, anime_df, mathias_anime_ratings5_ids)
    mathias_recom.to_csv('results5.csv')
    print(mathias_recom)


if __name__ == "__main__":

    main()
