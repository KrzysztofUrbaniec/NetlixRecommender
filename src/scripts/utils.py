import numpy as np
import pandas as pd

from surprise.model_selection import GridSearchCV
from surprise.model_selection import LeaveOneOut

from scripts.Metrics import Metrics

# ----------------------------------------------- Exploration.ipynb ----------------------------------------------
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

def print_sample_summary(sample):
    print(f"Actual number of unique users in the sample: {sample['CustId'].unique().size}")
    print(f"Actual number of unique movies in the sample: {sample['MovieId'].unique().size}")
    print(f"Total number of ratings in the sample: {len(sample)}")

# ----------------------------------------------- Modeling.ipynb ----------------------------------------------

def evaluate_baseline_performance(algo, train, test, verbose=True):

    if verbose is True: print('Training the model and making predictions...')
    algo.fit(train)
    predictions = algo.test(test)

    if verbose is True: print('Computing RMSE and MAE...')
    rmse_test = Metrics.RMSE(predictions, verbose=verbose)
    mae_test = Metrics.MAE(predictions, verbose=verbose)
    
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
    results_df = pd.DataFrame(results, index=['RMSE', 'MAE']).T.sort_values(by='RMSE').round(4) 
    # best_algo = algorithms[results_df.iloc[0].name]
    return results_df

def grid_search(data, algo_class, param_grid, measures=['rmse','mae'], cv=3, n_jobs=-1, joblib_verbose=3):
    
    grid_search = GridSearchCV(algo_class, param_grid, measures=measures, cv=cv, n_jobs=n_jobs, joblib_verbose=joblib_verbose)

    print('Initializing the search...')
    grid_search.fit(data)

    print('Extracting best parameters and scores...')
    print('\nDone.')
    return grid_search.best_params, grid_search.best_score

def predict_top_n_recs(algo, ratings, n=10, random_state=0):
    LOOCV = LeaveOneOut(n_splits=1, random_state=random_state)
    for train, test in LOOCV.split(ratings):
        algo.fit(train)
        leftOutPredictions = algo.test(test) 
        bigTestSet = train.build_anti_testset()
        allPredictions = algo.test(bigTestSet)
        topN = Metrics.GetTopN(allPredictions, n=n)
    return leftOutPredictions, topN

def compute_topN_metrics(left_out_predictions, topN, popranks, simsAlgo, n_users, print_metrics=True):
    metrics = dict()
    metrics['HR'] = Metrics.HitRate(topN, left_out_predictions)
    metrics['cHR'] = Metrics.CumulativeHitRate(topN, left_out_predictions, ratingCutoff=4.0)
    metrics['ARHR'] = Metrics.AverageReciprocalHitRank(topN, left_out_predictions)
    metrics['Diversity'] = Metrics.Diversity(topN, simsAlgo)
    metrics['User Coverage'] = Metrics.UserCoverage(topN, n_users, ratingThreshold=4.0)
    metrics['Novelty'] = Metrics.Novelty(topN, popranks)
    
    if print_metrics:
        for metric, value in metrics.items():
            print(f'{metric} = {value:4f}')

        print("\nLegend:\n")    
        print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
        print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
        print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better." )
        print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
        print("           for a given user. Higher means more diverse.")
        print("User Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
        print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")

    return metrics