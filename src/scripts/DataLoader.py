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

        # Dataset for loading data files
        dataset = Dataset.load_from_file(self.ratings_filepath, reader)

        with open(self.movies_filepath) as movies:
            csv_reader = csv.reader(movies, delimiter=',')
            next(csv_reader)
            for line in csv_reader:
                movieID = int(line[0])
                movie_name = line[2]
                self.movieID_to_name[movieID] = movie_name 
        
        for movieID, name in self.movieID_to_name.items():
            self.name_to_movieID[name] = movieID

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

    def get_movie_name(self, movie_id):
        return self.movieID_to_name[movie_id]

    def get_movie_id(self, movie_name):
        return self.name_to_movieID[movie_name]
