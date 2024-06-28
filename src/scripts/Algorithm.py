from scripts.Metrics import Metrics

class Algorithm:

    def __init__(self, algo, dataclass, popularity_ranks):
        self.dataclass = dataclass
        self.algo = algo
        self.popularity_ranks = popularity_ranks
    
    def evaluate(self, topN=10, print_metrics=True):
        self.metrics = {}

        self.algo.fit(self.dataclass.get_LOOCV_trainset())
        self.leftOutPredictions = self.algo.test(self.dataclass.get_LOOCV_testset()) 
        allPredictions = self.algo.test(self.dataclass.get_anti_testset())
        topN = Metrics.GetTopN(allPredictions, n=topN)

        self.metrics['HR'] = Metrics.HitRate(topN, self.left_out_predictions)
        self.metrics['cHR'] = Metrics.CumulativeHitRate(topN, self.left_out_predictions, ratingCutoff=4.0)
        self.metrics['ARHR'] = Metrics.AverageReciprocalHitRank(topN, self.left_out_predictions)
        self.metrics['Diversity'] = Metrics.Diversity(topN, self.dataclass.get_similarity_baseline_algo())
        self.metrics['User Coverage'] = Metrics.UserCoverage(topN, self.dataclass.get_n_users(), ratingThreshold=4.0)
        self.metrics['Novelty'] = Metrics.Novelty(topN, self.popularity_ranks)
        
        if print_metrics:
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

    # For user
    def recommend_top_n():
        pass