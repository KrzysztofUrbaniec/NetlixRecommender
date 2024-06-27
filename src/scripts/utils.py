import numpy as np
import pandas as pd

from surprise.accuracy import rmse, mae

def get_sample_data(ratings, n_users, n_movies, random_state=0):
    np.random.seed(random_state)

    # Get an array of unique users/movies
    unique_users = np.unique(ratings['CustId'])
    unique_movies = list(np.unique(ratings['MovieId']))

    # Check if requested number of users and movies is not more than available
    if n_users > len(unique_users):
        raise ValueError(f"Requested number of users {n_users} exceeds available {len(unique_users)} unique users.")
    if n_movies > len(unique_movies):
        raise ValueError(f"Requested number of movies {n_movies} exceeds available {len(unique_movies)} unique movies.")

    # Randomly select a subset of users/movies without replacement
    users_sample = np.random.choice(unique_users, size=n_users, replace=False)
    movies_sample = np.random.choice(unique_movies, size=n_movies, replace=False)

    # Filter the dataset with respect to the desired number of users
    sample = ratings[(ratings['CustId'].isin(users_sample)) & (ratings['MovieId'].isin(movies_sample))]

    return sample

def evaluate_baseline_performance(algo, train, test, verbose=True):

    if verbose is True: print('Training the model and making predictions...')
    algo.fit(train)
    predictions = algo.test(test)

    if verbose is True: print('Computing RMSE and MAE...')
    rmse_test = rmse(predictions, verbose=True)
    mae_test = mae(predictions, verbose=True)
    
    if verbose is True: print('\nDone.')

    return rmse_test, mae_test

def test_algorithms(algorithms: dict, train, test, verbose=True):
    # Dictionary to store the results    
    results = {}
    for (name, algo) in algorithms.items():
        print(f'Evaluating performance for: {name}')
        rmse_algo, mae_algo = evaluate_baseline_performance(algo, train, test, verbose=verbose)
        results[name] = [rmse_algo, mae_algo]
        print('')
    return pd.DataFrame(results, index=['RMSE', 'MAE']).T.sort_values(by='RMSE').round(4) 