'''Advanced Topic in Controls: Largew Scale Convex Optimization. Programming Exercise 10 '''

__author__ = 'Manuel Galliker'
__license__ = 'GPL'

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import math


def generate_poketype_matrix(poketype_list):
    # Task 1a)
    # generates a selection matrix to assign the correct types to all pokemons
    pokedata_frame = pd.read_csv('pokemon.csv')
    poketype_frame = pokedata_frame[['type1', 'type2']]
    poketype_matrix = pd.DataFrame(np.zeros(
        (poketype_frame.shape[0], len(poketype_list))), columns=poketype_list)
    for row in range(poketype_matrix.shape[0]):
        for current_type in poketype_list:
            if (poketype_frame.loc[row, 'type1'] == current_type or poketype_frame.loc[row, 'type2'] == current_type):
                poketype_matrix.loc[row, current_type] = 1
    return poketype_matrix, poketype_matrix.shape[0]


def generate_pokemoves_matrix(pokemon_count):
    # Task 1b)
    # generates a selection matrix to assign the possible moves to each pokemon
    pokedata_frame = pd.read_csv('pokemon_moves.csv')
    pokedata_frame = pokedata_frame[pokedata_frame['pokemon_id'].between(
        1, 801, inclusive=True)]
    pokemoves_frame = pokedata_frame[['pokemon_id', 'move_id']]
    pokemoves_matrix = pd.DataFrame(
        np.zeros((728, pokemon_count)))
    for row in range(pokemoves_frame.shape[0]):
        current_pokemon = pokemoves_frame.loc[row, 'pokemon_id'] - 1
        current_move = pokemoves_frame.loc[row, 'move_id'] - 1
        pokemoves_matrix.iloc[current_move, current_pokemon] = 1

    return pokemoves_matrix


def booleanize_classifier(classifier):
    for i in range(classifier.shape[0]):
        for j in range(classifier.shape[1]):
            if(classifier[i, j] > 0.0):
                classifier[i, j] = 1
            else:
                classifier[i, j] = 0
    return classifier


def evaluate_classifier(estimated_type_classifier, type_classifier, poketype_list):

    if (estimated_type_classifier.shape != type_classifier.shape):
        print('ERROR: Try to evaluate matrices of different size')
        print('estimated_type_classifier.shape:',
              estimated_type_classifier.shape)
        print('type_classifier.shape:', type_classifier.shape)
        return

    results = pd.DataFrame(np.zeros((type_classifier.shape[1] + 1, 5)), columns=[
                           'Class Index', 'True Pos', 'True Neg', 'False Pos', 'False Neg'])
    estimated_type_classifier = np.atleast_2d(estimated_type_classifier)
    type_classifier = np.atleast_2d(type_classifier)

    for i in range(type_classifier.shape[0]):
        for j in range(type_classifier.shape[1]):
            if (estimated_type_classifier[i, j] == 0 and type_classifier[i, j] == 0):
                results.loc[j, 'True Neg'] += 1
            elif (estimated_type_classifier[i, j] == 1 and type_classifier[i, j] == 1):
                results.loc[j, 'True Pos'] += 1
            elif (estimated_type_classifier[i, j] == 1 and type_classifier[i, j] == 0):
                results.loc[j, 'False Pos'] += 1
            elif (estimated_type_classifier[i, j] == 0 and type_classifier[i, j] == 1):
                results.loc[j, 'False Neg'] += 1
            else:
                print(
                    'ERROR: evaluated matrices are not binary at pos:', [i, j])
    results.loc[type_classifier.shape[1],
                'True Neg'] = results['True Neg'].sum()
    results.loc[type_classifier.shape[1],
                'True Pos'] = results['True Pos'].sum()
    results.loc[type_classifier.shape[1],
                'False Pos'] = results['False Pos'].sum()
    results.loc[type_classifier.shape[1],
                'False Neg'] = results['False Neg'].sum()

    result_index_list = poketype_list + ['overall']
    results['Class Index'] = result_index_list
    # results.set_index('Class Index')
    return results

def compute_opt_beta(pokemoves_matrix, poketype_matrix,  poketype_list, n_train):
    m_mat = pokemoves_matrix.to_numpy()
    beta_mat = np.zeros((m_mat.shape[0], len(poketype_list)))

    for classifier_type in poketype_list:
        print('start optimization for type ',
              classifier_type, 'ntrain: ', n_train)
        y_vec = poketype_matrix[classifier_type].to_numpy()
        beta = cp.Variable(beta_mat.shape[0])
        objective = 0
        for i in range(n_train):
            objective += y_vec[i] * m_mat[:, i].T @ beta - cp.logistic(m_mat[:, i].T @ beta)

        prob = cp.Problem(cp.Maximize(objective))
        prob.solve(verbose=False, warm_start=True, solver=cp.ECOS)
        # print("Optimal var reached", beta.value)
        beta_mat[:, poketype_list.index(classifier_type)] = beta.value

    return beta_mat

def compute_test_classifier(pokemoves_matrix, poketype_matrix,  poketype_list, n_train):

    m_mat = pokemoves_matrix.to_numpy()
    beta_mat = np.zeros((m_mat.shape[0], len(poketype_list)))

    for classifier_type in poketype_list:
        print('start optimization for type ',
              classifier_type, 'ntrain: ', n_train)
        y_vec = poketype_matrix[classifier_type].to_numpy()
        beta = cp.Variable(beta_mat.shape[0])
        objective = 0
        for i in range(n_train):
            objective += y_vec[i] * m_mat[:, i].T @ beta - cp.logistic(m_mat[:, i].T @ beta)

        prob = cp.Problem(cp.Maximize(objective))
        prob.solve(verbose=False, warm_start=True, solver=cp.ECOS)
        # print("Optimal var reached", beta.value)
        beta_mat[:, poketype_list.index(classifier_type)] = beta.value

    m_mat_test = m_mat[:, range(n_train, m_mat.shape[1])]
    test_classifier = booleanize_classifier(m_mat_test.T @ beta_mat)
    classifier_results = evaluate_classifier(test_classifier, poketype_matrix.iloc[range(
        n_train, m_mat.shape[1]), :].to_numpy(), poketype_list)
    return test_classifier, classifier_results


def add_balanced_accuracy(classifier_results):
    classifier_results['Selectivity'] = classifier_results['True Pos'] / \
        (classifier_results['True Pos'] + classifier_results['False Neg'])
    classifier_results['Specivity'] = classifier_results['True Neg'] / \
        (classifier_results['True Neg'] + classifier_results['False Pos'])
    classifier_results['Balanced Accuracy'] = (
        classifier_results['Specivity'] + classifier_results['Selectivity'])/2
    return classifier_results


def add_proposed_ratios(classifier_results):
    classifier_results['r1'] = classifier_results['False Pos'] / \
        (classifier_results['True Pos'])
    classifier_results['r2'] = classifier_results['False Neg'] / \
        (classifier_results['True Neg'])
    classifier_results['r3'] = (classifier_results['False Pos'] + classifier_results['False Neg'])/(
        classifier_results['True Pos'] + classifier_results['True Neg'])
    return classifier_results


def generate_plot(ratio_type, poketype_list, classifier_res_container, n_train):
    fig1, ax = plt.subplots()
    for type_counter in range(len(poketype_list)):
        r = []
        for i in range(len(n_train)):
            cur_df = classifier_res_container[i]
            r.append(cur_df.loc[type_counter, ratio_type])
        ax.plot(n_train, r, label=poketype_list[type_counter])

    ax.set(xlabel='training sample size', ylabel=ratio_type,
           title=ratio_type)
    ax.legend(loc='upper right')
    ax.grid()
    fig1.savefig(ratio_type + ".png")
    plt.show()


def main():

    print(cp.installed_solvers())
    poketype_list = ['normal', 'fire', 'fighting', 'water', 'flying', 'grass', 'poison', 'electric',
                     'ground', 'psychic', 'rock', 'ice', 'bug', 'dragon', 'ghost', 'dark', 'steel', 'fairy']

    # # task 1a)
    # poketype_matrix, pokemon_count = generate_poketype_matrix(poketype_list)
    # poketype_matrix.to_csv('poketype_matrix_1a.csv')

    # # task 1b)
    # pokemoves_matrix = generate_pokemoves_matrix(pokemon_count)
    # pokemoves_matrix.to_csv('pokemoves_matrix_1b.csv')

    poketype_matrix = pd.read_csv('poketype_matrix_1a.csv')
    pokemoves_matrix = pd.read_csv('pokemoves_matrix_1b.csv')

    # Test matrices with the provided ones
    provided_poketype_matrix = pd.read_csv(
        'ex10_pokemon_types.csv', header=None)
    provided_pokemoves_matrix = pd.read_csv(
        'ex10_pokemon_moves.csv', header=None)
    if(provided_poketype_matrix.values.any() != poketype_matrix.values.any()):
        print('ERROR: Poketype matrix not identical!')
    else:
        print('poketype_matrix check passed!')
    if(provided_pokemoves_matrix.values.any() != pokemoves_matrix.values.any()):
        print('ERROR: Pokemove matrix not identical!')
    else:
        print('pokemove_matrix check passed!')

    # task 1c) & 1d)
    test_classifier, classifier_results = compute_test_classifier(
        pokemoves_matrix, poketype_matrix, poketype_list, 50)
    print(classifier_results)
    np.savetxt('classifier_1d.csv', test_classifier, delimiter=',')
    classifier_results.to_csv('classifier_results_1d.csv')

    # # task 1e)
    # balanced_results = add_balanced_accuracy(classifier_results)
    # print(balanced_results)
    # balanced_results.to_csv('balanced_classifier_results_1e.csv')

    # # task 1f)
    # n_train = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    # ratio_classifier_results = []
    # balanced_classifier_results = []
    # for n in n_train:
    #     test_classifier, classifier_results = compute_test_classifier(
    #         pokemoves_matrix, poketype_matrix, poketype_list, n)
    #     cur_ratio_classifier_results = add_proposed_ratios(classifier_results)
    #     ratio_classifier_results.append(cur_ratio_classifier_results)
    #     cur_balanced_classifier_results = add_balanced_accuracy(classifier_results)
    #     balanced_classifier_results.append(cur_balanced_classifier_results)

    # print(ratio_classifier_results)

    # generate_plot('r1', poketype_list, ratio_classifier_results, n_train)
    # generate_plot('r2', poketype_list, ratio_classifier_results, n_train)
    # generate_plot('r3', poketype_list, ratio_classifier_results, n_train)

    # generate_plot('Selectivity', poketype_list, ratio_classifier_results, n_train)
    # generate_plot('Specivity', poketype_list, ratio_classifier_results, n_train)
    # generate_plot('Balanced Accuracy', poketype_list, ratio_classifier_results, n_train)

    # task 1g)
    # With the same classifier for single and double pokemon:
    test_poketype_matrix = poketype_matrix.iloc[range(400, 801), :]
    test_pokemoves_matrix = pokemoves_matrix.iloc[:, range(400, 801)]

    single_poke_id_list = []
    double_poke_id_list = []
    for i in range(test_poketype_matrix.shape[0]):
        print(i)
        cur_row = test_poketype_matrix.iloc[i, range(1,19)].to_numpy()
        if (cur_row.sum() == 1):
            single_poke_id_list.append(i)
        elif (cur_row.sum() == 2):
            double_poke_id_list.append(i)
    print(single_poke_id_list)
    print(double_poke_id_list)

    single_test_poketype_matrix = test_poketype_matrix.iloc[single_poke_id_list, range(1,19)].to_numpy()
    single_test_pokemoves_matrix = pokemoves_matrix.iloc[:, single_poke_id_list].to_numpy()
    print(single_test_poketype_matrix)
    print(single_test_poketype_matrix.shape)

    double_test_poketype_matrix = test_poketype_matrix.iloc[double_poke_id_list, range(1,19)].to_numpy()
    double_test_pokemoves_matrix = pokemoves_matrix.iloc[:, double_poke_id_list].to_numpy()
    print(double_test_poketype_matrix)
    print(double_test_poketype_matrix.shape)

    beta_mat = compute_opt_beta(
        pokemoves_matrix, poketype_matrix, poketype_list, 400)


    single_classifier = booleanize_classifier(single_test_pokemoves_matrix.T @ beta_mat)
    single_classifier_results = evaluate_classifier(single_classifier, single_test_poketype_matrix, poketype_list)
    single_classifier_results = add_balanced_accuracy(single_classifier_results)

    double_classifier = booleanize_classifier(double_test_pokemoves_matrix.T @ beta_mat)
    double_classifier_results = evaluate_classifier(double_classifier, double_test_poketype_matrix, poketype_list)
    double_classifier_results = add_balanced_accuracy(double_classifier_results)

    single_classifier_results.to_csv('single_classifier_single_beta_results.csv')
    double_classifier_results.to_csv('double_classifier_single_beat results.csv')

    print(single_classifier_results)
    print(double_classifier_results)



    # # With different classifiers for single and double pokemon:
    # single_poke_id_list = []
    # double_poke_id_list = []
    # for i in range(poketype_matrix.shape[0]):
    #     cur_row = poketype_matrix.iloc[i, :].to_numpy()
    #     if (cur_row.sum() == 1):
    #         single_poke_id_list.append(i)
    #     elif (cur_row.sum() == 2):
    #         double_poke_id_list.append(i)

    # single_pokemoves_matrix = pokemoves_matrix.iloc[:, single_poke_id_list]
    # single_poketype_matrix = poketype_matrix.iloc[single_poke_id_list, :]

    # double_pokemoves_matrix = pokemoves_matrix.iloc[:, double_poke_id_list]
    # double_poketype_matrix = poketype_matrix.iloc[double_poke_id_list, :]

    # print(double_poketype_matrix.shape)
    # print(single_poketype_matrix.shape)

    # single_classifier, single_classifier_results = compute_test_classifier(
    #     single_pokemoves_matrix, single_poketype_matrix, poketype_list, 200)
    # double_classifier, double_classifier_results = compute_test_classifier(
    #     double_pokemoves_matrix, double_poketype_matrix, poketype_list, 200)

    # single_classifier_results = add_balanced_accuracy(
    #     single_classifier_results)
    # double_classifier_results = add_balanced_accuracy(
    #     double_classifier_results)

    # print(single_classifier_results)
    # print(double_classifier_results)

    # single_classifier

    # single_classifier_results.to_csv('single_classifier_double_beta_results.csv')
    # double_classifier_results.to_csv('double_classifier__double_beat results.csv')

    return


if __name__ == "__main__":

    main()
