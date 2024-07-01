import numpy as np
import pandas as pd

class MovieSampler:

    def __init__(self, ratings):
        self.ratings = ratings

    def compute_weights(self, n_groups, s=1.0):
        '''Compute normalized weights using Zipf-like distribution. 
        
        The parameter s controls the proportion of samples drawn from groups with the highest number of ratings. 
        The higher its value, the more samples are drawn from these top-rated groups.'''

        # Use zipfan distribution
        ranks = np.arange(1, n_groups + 1)
        weights = 1 / np.power(ranks, s)
        total_weight = np.sum(weights)
        normalized_weights = weights / total_weight
        return normalized_weights

    def adjust_counts(self, weights, total_count):
        '''"Adjust number of samples in particular groups to ensure the sum matches the desired total count.'''
        initial_counts = np.round(weights * total_count).astype(int) 
        residuals = (weights * total_count) - initial_counts
        difference = total_count - np.sum(initial_counts)
        
        # Redistribute the difference
        indices = np.argsort(residuals)[::-1]
        for i in range(np.abs(difference)):
            # j = -1 if np.sign(difference) < 0 else +1 
            initial_counts[indices[i]] += np.sign(difference) # Add or remove items depending on the sign of the difference
            if initial_counts.sum() == total_count: break
        
        return initial_counts

    def get_movie_ranks(self, ratings):
        '''Prepare movie ranks (lowest ranks correspond to the highest number of ratings).'''
        n_ratings_per_movie = ratings.groupby('MovieId')['CustId'].count().sort_values(ascending=False).to_frame().rename({'CustId':'count'},axis=1)
        n_ratings_per_movie['rank'] = n_ratings_per_movie['count'].rank(ascending=False, axis=0)
        return n_ratings_per_movie

    def divide_movies_into_groups(self, ranks_ratings_df, n_groups):
        '''Divide the dataset into disjoint groups based on rank percentiles.'''
        groups = pd.qcut(ranks_ratings_df['rank'], q=n_groups, labels=np.arange(1, n_groups+1))
        return groups

    def sample_movies(self, groups, n_movies_per_group):
        '''Sample movies from the groups based on specified distribution.'''
        selected_movie_idx = {}
        for group_idx in np.unique(groups.to_numpy()):
            # Select indices of all movies in a group
            movie_idx_in_group = groups[groups == group_idx].index.to_numpy()
            # Randomly select a subset of movies
            if len(movie_idx_in_group) > n_movies_per_group[group_idx - 1]:
                selected_movies_in_group = np.random.choice(movie_idx_in_group, n_movies_per_group[group_idx - 1], replace=False)
            else:
                selected_movies_in_group = movie_idx_in_group.copy()
            
            selected_movie_idx[group_idx] = selected_movies_in_group
        return selected_movie_idx

    def sample_users(self, groups, n_users_per_group, selected_movies_per_group):
        '''Sample users from the groups based on specified distribution.'''

        # Why start with the last group?
        # The last group contains movies with the least number of ratings

        # The algorithm selects user ids in each group randomly, but also removes all ids,
        # which were selected in earlier groups, so they don't get duplicated

        # The first group provides the largest number of unique users,
        # and if we started sampling with this group, then, 
        # due to id removal mentioned above, these ids couldn't 
        # be selected in the next groups and the movies 
        # they contain would have lower number of ratings

        # This could lead to a situation, where there are too few unique
        # user ids for movies in last groups (because they were selected
        # earlier for more popular movies), which means, that the desired
        # sample size would not be attainable.

        # This in turn means, that we should start with the last group
        # (least popular movies) and move towards the first one (most popular movies)

        selected_user_idx = []
        for group_idx in np.unique(groups.to_numpy())[::-1]:
            # Find unique users, who watched the movies in the group and remove ids, which have already been selected
            unique_users_in_group = self.ratings[self.ratings['MovieId'].isin(selected_movies_per_group[group_idx])]['CustId'].unique()
            unique_users_in_group = np.setdiff1d(unique_users_in_group,selected_user_idx)

            # Randomly draw n_users_per_group users from filtered array of indices
            # If the number of users in the group determined earlier is greater than 
            # the number of unique users in the group, select all unique users
            n_users = min(n_users_per_group[group_idx - 1], unique_users_in_group.size)
            randomly_selected_users = np.random.choice(unique_users_in_group, size=n_users, replace=False)
            selected_user_idx += randomly_selected_users.tolist()
        return selected_user_idx

    def get_sample_data(self, n_users, n_movies, n_groups=10, random_state=0, zipfs_s=1.0):
        '''
        Generate a subset of the dataset with a specified number of users and movies, 
        divided into groups based on movie ranks. 

        Parameters:
        -----------
        ratings : pd.DataFrame
            DataFrame containing movie ratings with columns ['CustId', 'MovieId', 'Rating', 'Date'].
        n_users : int
            The number of unique users to sample.
        n_movies : int
            The number of unique movies to sample.
        n_groups : int, optional, default=10
            The number of groups to divide the movies into based on their rank percentiles.
        random_state : int, optional, default=0
            Random seed for reproducibility.
        user_weights_s : float, optional, default=1.0
            Parameter to control the proportion of samples from groups with the highest number of ratings.
            Higher values result in more samples drawn from these groups.

        Returns:
        --------
        pd.DataFrame
            Subset of the ratings DataFrame containing the selected users and movies.
        '''
        
        # NOTE: The algorithm in current form will work correcly only if there is enough ranks to divide movies into groups (group edges don't overlap)
        # TODO: DataFrame columns are currently accessed with keys specific to Netlifx Prize dataset. 
        # TODO: Either access the columns with numbers and assume specific order or request that they have some particular names
        # TODO2: The number of requested movies can alter the actual number of sampled unique users and vice verse
        # TODO2: For example, if the number of movies is too low, there may be not enough unique users to satisfy the requirement
        # TODO2: Consider addind a warning for the user, which explains this issue

        np.random.seed(random_state)

        # Prepare movie ranks
        n_ratings_per_movie = self.get_movie_ranks(self.ratings)

        # Use rank percentiles to divide the movies into groups 
        # TODO3: Devise a solution, that will allow to divide the movies if there is not enough distinct ranks or prepare default behavior for such case
        groups = self.divide_movies_into_groups(n_ratings_per_movie, n_groups)

        # Compute number of users to sample from each rank group using Zipf's law
        # The same law is used to compute the number of movies (TODO: Consider using geometric distribution or even uniform sampling)    
        weights = self.compute_weights(n_groups, s=zipfs_s) 
        n_users_per_group = self.adjust_counts(weights, n_users) # Zipf's distribution is (probably) justified
        n_movies_per_group = self.adjust_counts(weights, n_movies) # How to correctly sample the movies?

        # Sample users and movies
        selected_movie_idx = self.sample_movies(groups, n_movies_per_group)
        movies_idx_concat = np.concatenate(list(selected_movie_idx.values()))
        selected_user_idx = self.sample_users(groups, n_users_per_group, selected_movie_idx)
        
        # Select a subset of ratings
        ratings_subset = self.ratings[(self.ratings['CustId'].isin(selected_user_idx)) & (self.ratings['MovieId'].isin(movies_idx_concat))]
        return ratings_subset