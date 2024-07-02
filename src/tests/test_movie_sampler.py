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
        
    # Check if the weights sum to 1 
    def test_normalization(self):
        self.assertAlmostEqual(np.sum(self.weights), 1.0)
        
    # Check if the weights are in descending order
    def test_descending_order(self):
        self.assertTrue(np.all(np.diff(self.weights) <= 0))

class TestAdjustCounts(BaseTestClass):

    def setUp(self):
        super().setUp()
        self.weights = np.array([0.521,0.233,0.182,0.064])

    # Check the returned (adjusted) counts
    def test_returned_counts(self):
        total_count = 100
        counts = self.ms.adjust_counts(self.weights, total_count)
        expected_counts = np.array([52,23,18,7])
        assert np.allclose(counts, expected_counts)

    # Check if the number of adjusted counts matches the total count
    def test_total_number_of_counts(self):
        total_count = 100
        counts = self.ms.adjust_counts(self.weights, total_count)
        self.assertAlmostEqual(counts.sum(), total_count)
        
    


    # def test_returned_n_users(self):
    #     test_cases = [
    #         (1000, 200, 5, 1000),
    #         (999, 200, 5, 999),
    #         (1, 200, 5, 1),
    #         (100_000, 200, 5, 100_000),
    #         (1000, 200, 10, 1000),
    #         (999, 200, 10, 999),
    #         (1, 200, 10, 1),
    #         (100_000, 200, 10, 100_000),
    #     ]

    #     for n_users, n_movies, n_groups, expected_unique_users in test_cases:
    #         with self.subTest(n_users=n_users, n_movies=n_movies, n_groups=n_groups):
    #             df = get_sample_data(self.ratings, n_users=n_users, n_movies=n_movies, n_groups=n_groups, random_state=self.random_state)
    #             self.assertEqual(df['CustId'].nunique(), expected_unique_users)

if __name__ == "__main__":
    unittest.main()