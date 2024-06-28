import numpy as np

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class Data:

    def __init__(self, data):
        self.data = data
        # Data is rankings

        self.full_trainset = data.build_full_trainset()
        
        LOOCV = LeaveOneOut(n_splits=1, random_state=0)
        
        # LOOCV train/test + antitestset
        for train, test in LOOCV.split(self.data):
            self.LOOCV_train = train
            self.LOOCV_test = test

        self.anti_testset = self.LOOCV_train.build_anti_testset()

        # Similarity matrix
        sim_options = {'name': 'cosine', 'user_based': False}
        self.simsAlgo = KNNBaseline(sim_options=sim_options, verbose=False)
        self.simsAlgo.fit(self.full_trainset)

        # Number of users and movies
        self.n_users = len(np.unique([int(row[0]) for row in self.data.raw_ratings]))
        self.n_movies = len(np.unique([int(row[2]) for row in self.data.raw_ratings]))

    # --------------- Getters --------------------

    def get_full_trainset(self):
        return self.full_trainset
    
    def get_LOOCV_trainset(self):
        return self.LOOCV_train
    
    def get_LOOCV_testset(self):
        return self.LOOCV_test
    
    def get_anti_testset(self):
        return self.anti_testset
    
    def get_n_users(self):
        return self.n_users
    
    def get_n_movies(self):
        return self.n_movies
    
    def get_similarity_baseline_algo(self):
        return self.simsAlgo