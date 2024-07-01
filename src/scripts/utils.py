import numpy as np
import pandas as pd

from surprise.model_selection import GridSearchCV

from src.scripts.Metrics import Metrics

# ----------------------------------------------- Exploration.ipynb ----------------------------------------------

def print_sample_summary(sample):
    print(f"Actual number of unique users in the sample: {sample['CustId'].nunique()}")
    print(f"Actual number of unique movies in the sample: {sample['MovieId'].nunique()}")
    print(f"Total number of ratings in the sample: {len(sample)}")

# ----------------------------------------------- Modeling.ipynb ----------------------------------------------

def evaluate_baseline_performance(algo, data, verbose=True):

    if verbose is True: print('Training the model and making predictions...')
    algo.fit(data.get_accuracy_trainset())
    predictions = algo.test(data.get_accuracy_testset())

    if verbose is True: print('Computing RMSE and MAE...')
    rmse_test = Metrics.RMSE(predictions, verbose=verbose)
    mae_test = Metrics.MAE(predictions, verbose=verbose)
    
    if verbose is True: print('\nDone.')

    return rmse_test, mae_test

def test_algorithms(algorithms: dict, data, verbose=True):
    # Dictionary to store the results    
    results = {}
    for (name, algo) in algorithms.items():
        print(f'Evaluating performance for: {name}')
        rmse_algo, mae_algo = evaluate_baseline_performance(algo, data, verbose=verbose)
        results[name] = [rmse_algo, mae_algo]
        print('')
    results_df = pd.DataFrame(results, index=['RMSE', 'MAE']).T.sort_values(by='RMSE').round(4) 
    # best_algo = algorithms[results_df.iloc[0].name]
    return results_df

def grid_search(data, algo_class, param_grid, measures=['rmse','mae'], cv=3, n_jobs=-1, joblib_verbose=3):
    
    grid_search = GridSearchCV(algo_class, param_grid, measures=measures, cv=cv, n_jobs=n_jobs, joblib_verbose=joblib_verbose)

    print('Initializing the search...')
    # Unfortunately, surprise, unlike sklearn, does not allow to print detailed info about the search
    grid_search.fit(data)

    print('Extracting best parameters and scores...')
    print('\nDone.')
    return grid_search.best_params, grid_search.best_score