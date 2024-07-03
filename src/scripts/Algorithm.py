'''This module provides a class designed to facilitate training and validation of a model.'''

from src.scripts.Metrics import Metrics

class Algorithm:

    def __init__(self, algo, dataclass, popularity_ranks):
        self.dataclass = dataclass
        self.algo = algo
        self.popularity_ranks = popularity_ranks
        self.fit_on_full_trainset = False
        self.evaluated = False

    def compute_accuracy(self):
        self.algo.fit(self.dataclass.get_accuracy_trainset())
        predictions = self.algo.test(self.dataclass.get_accuracy_testset())
        rmse = Metrics.RMSE(predictions, verbose=False)
        mse = Metrics.MAE(predictions, verbose=False)
        return rmse, mse
    
    def create_testset_surprise(self, ratings_test):
        '''
        Prepare a testset in the format required by surprise.
        
        Parameters:
        -----------
        ratings_test: pd.DataFrame
            Pandas DataFrame with three columns: (user_id, rating, movie_id)
        '''
        testset = [(user_id, movie_id, rating) for (user_id, rating, movie_id) in ratings_test.to_numpy()]
        return testset
    
    def evaluate(self, topN=10, compute_accuracy_only=True, verbose=True):
        '''Evaluate the model with basic accuracy metrics (rmse, mse) and optional user-centric metrics.
        
        Parameters:
        ----------
        topN: int, optional, default=10
            Number of recommendations made for each user in leave-one-out CV testset.
        compute_accuracy_only: bool, optional, default=True
            A boolean flag to indicate whether to compute user-centric metrics in addition to accuracy metrics.
        verbose: bool, optional, default=True
            A boolean flag to indicate whether to print the additional logs.
        '''

        self.metrics = {}
        self.evaluated = True

        if verbose: print('Computing accuracy metrics...')
        (self.metrics['RMSE'], self.metrics['MSE']) = self.compute_accuracy()
        
        if compute_accuracy_only is False:
            if verbose: 
                print('\nComputing user-centered metrics...')
                print('Fitting the algorithm to the leave-one-out CV trainset...')
            self.algo.fit(self.dataclass.get_LOOCV_trainset())
            if verbose: print('Making predictions for the leave-one-out CV testset...')
            self.left_out_predictions = self.algo.test(self.dataclass.get_LOOCV_testset()) 
            if verbose: print('Making precictions for the anti testset...')
            all_predictions = self.algo.test(self.dataclass.get_anti_testset())
            if verbose: print('Generating top N recommendations for all users...')
            topN = Metrics.GetTopN(all_predictions, n=topN, minimumRating=4.0)

            if verbose: print('Computing hit rate...')
            self.metrics['HR'] = Metrics.HitRate(topN, self.left_out_predictions)
            if verbose: print('Computing cumulative hit rate...')
            self.metrics['cHR'] = Metrics.CumulativeHitRate(topN, self.left_out_predictions, ratingCutoff=4.0)
            if verbose: print('Computing average reciprocal hit rank...')
            self.metrics['ARHR'] = Metrics.AverageReciprocalHitRank(topN, self.left_out_predictions)
            if verbose: print('Computing diversity...')
            self.metrics['Diversity'] = Metrics.Diversity(topN, self.dataclass.get_similarity_baseline_algo())
            if verbose: print('Computing user coverage...')
            self.metrics['User Coverage'] = Metrics.UserCoverage(topN, self.dataclass.get_n_users(), ratingThreshold=4.0)
            if verbose: print('Computing novelty...')
            self.metrics['Novelty'] = Metrics.Novelty(topN, self.popularity_ranks)

            print('')
            for metric, value in self.metrics.items():
                print(f'{metric} = {value:4f}')

            print("\nLegend:\n")    
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
            print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better." )
            print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("User Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")

    def fit_with_full_trainset(self):
        '''Fit the model on the full trainset. Use before generating recommendations or validating the model with a testset.'''
        trainset = self.dataclass.get_full_trainset()
        self.algo.fit(trainset) 
        self.fit_on_full_trainset = True

    def generate_recommendations(self, user_id, topN=10):
        '''Generate a list of top N recommendations for specific user. '''
        if self.fit_on_full_trainset is False: 
            raise ValueError("Can't generate predictions. The model hasn't been fitted yet. Call fit_with_full_trainset() and retry.")
        predictions = self.algo.test(self.dataclass.get_anti_testset_for_user(user_id))
        recs = []
        
        # "test" method returns a list of predictions
        for _, movie_id, _, estimated_rating, _ in predictions:
            recs.append((int(movie_id), estimated_rating))

        # Sort descending and select top N
        return sorted(recs, reverse=True, key=lambda x: x[1])[:topN]
    
    def get_metrics(self):
        if self.evaluated is False:
            raise ValueError('There are no metrics yet. Evaluate the model first.')
        return self.metrics
    
    def get_algorithm(self):
        return self.algo
    
    def validate_test(self, testset_df):
        '''
        Validate the model using provided testset.

        Parameters: 
        -----------
        testset_df: pd.DataFrame
            Pandas DataFrame with three columns: (user_id, rating, movie_id)
        '''
        if self.fit_on_full_trainset is False: 
            raise ValueError("Can't generate predictions. The model hasn't been fitted yet. Call fit_with_full_trainset() and retry.")
        
        # Prepare testset in format required by surprise
        self.testset = self.create_testset_surprise(testset_df)

        # Generate predictions
        test_predictions = self.algo.test(self.testset)

        print(f'RMSE: {Metrics.RMSE(test_predictions, verbose=False)}')
        print(f'MAE: {Metrics.MAE(test_predictions, verbose=False)}')