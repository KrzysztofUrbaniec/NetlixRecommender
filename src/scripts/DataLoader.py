import csv

from collections import defaultdict

from surprise import Dataset
from surprise import Reader

class DataLoader:

    def __init__(self, ratings_filepath, movies_filepath):
        self.movieID_to_name = {}
        self.name_to_movieID = {}
        self.movies_filepath = movies_filepath
        self.ratings_filepath = ratings_filepath

    def load_dataset(self):
        # Reader class for parsing data files
        reader = Reader(line_format='user rating item', sep=',', rating_scale=(1,5), skip_lines=1)

        # Dataset for loading datasets
        dataset = Dataset.load_from_file(self.ratings_filepath, reader)

        return dataset

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratings_filepath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1
        rank = 1
        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1
        return rankings

    def get_movie_name(self):
        pass

    def get_movie_id(self):
        pass
