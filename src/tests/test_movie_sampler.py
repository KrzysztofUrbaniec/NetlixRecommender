'''This module provides some basic tests for MovieSampler class.'''

import unittest
import pandas as pd
import numpy as np

from src.scripts.MovieSampler import MovieSampler

class BaseTestClass(unittest.TestCase):

    def setUp(self):
        filepath = 'data/ratings_training.csv'
        ratings_dtypes = {'CustId':np.uint32, 'Rating':np.uint8, 'MovieId':np.uint32}
        self.ratings = pd.read_csv(filepath,dtype=ratings_dtypes)
        self.ms = MovieSampler(self.ratings)

        self.random_state = 42

class TestComputeWeights(BaseTestClass):

    def setUp(self):
        super().setUp()
        self.weights = self.ms.compute_weights(10, s=1.0, verbose=False)
        
    def test_normalization(self):
        '''Test if the weights are normalized (i.e. sum to 1).'''
        self.assertAlmostEqual(np.sum(self.weights), 1.0)

    def test_descending_order(self):
        '''Test if the weights are in descending order.'''
        self.assertTrue(np.all(np.diff(self.weights) <= 0))

class TestAdjustCounts(BaseTestClass):

    def setUp(self):
        super().setUp()
        self.weights = np.array([0.521,0.233,0.182,0.064])

    def test_returned_counts(self):
        '''Test if the returned counts match the expected counts.'''
        total_count = 100
        counts = self.ms.adjust_counts(self.weights, total_count)
        expected_counts = np.array([52,23,18,7])
        self.assertTrue(np.allclose(counts, expected_counts))

    def test_total_number_of_counts(self):
        '''Test if the sum of adjusted counts matches the total count.'''
        total_count = 100
        counts = self.ms.adjust_counts(self.weights, total_count)
        self.assertAlmostEqual(counts.sum(), total_count)

class TestDivideMoviesIntoGroups(BaseTestClass):

    def setUp(self):
        super().setUp()
        self.ranks = self.ms.get_movie_ranks(self.ratings, verbose=False)
        
    def test_nunique_groups(self):
        """Test if the number of unique groups matches the specified number."""
        for n_groups in [10, 5]:
            with self.subTest(n_groups=n_groups):
                unique_groups = self.ms.divide_movies_into_groups(self.ranks, n_groups=n_groups, verbose=False).nunique()
                self.assertEqual(unique_groups, n_groups)

class TestSampleMovies(BaseTestClass):
    
    def setUp(self):
        super().setUp()
        self.ranks = self.ms.get_movie_ranks(self.ratings, verbose=False)
        self.weights = self.ms.compute_weights(10, s=1.0, verbose=False)

        self.n_groups = 10

    def test_total_movie_count(self):
        '''Test if the total number of sampled movies matches the expected count.'''
        test_cases = [99,100,101,152]

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                self.n_movies_per_group = self.ms.adjust_counts(self.weights, total_count=test_case)
                self.groups = self.ms.divide_movies_into_groups(self.ranks, n_groups=self.n_groups, verbose=False)
                # Movie_groups is dictionary, where keys are group indices and values are arrays with selected movies
                self.movie_groups = self.ms.sample_movies(self.groups, self.n_movies_per_group, verbose=False)

                count = np.concatenate(list(self.movie_groups.values())).size
                expected_count = test_case
                self.assertEqual(count, expected_count)


class TestSampleUsers(BaseTestClass):
   
    def setUp(self):
        super().setUp()
        self.ranks = self.ms.get_movie_ranks(self.ratings, verbose=False)
        self.weights = self.ms.compute_weights(10, s=1.0, verbose=False)

        self.n_groups = 10
        self.movie_total_count = 200

    def test_total_user_count(self):
        '''Test if the total number of sampled users matches the expected count.'''
        test_cases = [999,1000,1001,1052]

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                n_users_per_group = self.ms.adjust_counts(self.weights, total_count=test_case)
                groups = self.ms.divide_movies_into_groups(self.ranks, n_groups=self.n_groups, verbose=False)
                n_movies_per_group = self.ms.adjust_counts(self.weights, total_count=self.movie_total_count)
                movie_groups = self.ms.sample_movies(groups, n_movies_per_group, verbose=False)
                sampled_users = self.ms.sample_users(groups, n_users_per_group, movie_groups, verbose=False) # A list of sampled userIds

                count = len(sampled_users)
                expected_count = test_case
                self.assertEqual(count, expected_count)

if __name__ == "__main__":
    unittest.main()