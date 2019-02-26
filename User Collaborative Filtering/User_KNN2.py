import numpy as np
from User_KNN import User_KNN

# Subclass of User_KNN, so only unique functionality needs to be implemented.
class User_KNN2(User_KNN):
    _shrinkage = 0
    _user_label = 'userId'
    _item_label = 'itemId'
    _rating_label = 'rating'

    def __init__(self, nhood_size, sim_threshold=0, shrinkage=0,
                 user_label='user', item_label='item', rating_label='score'):
        User_KNN.__init__(self, nhood_size, sim_threshold=sim_threshold)

        self._shrinkage = shrinkage
        self._user_label = user_label
        self._item_label = item_label
        self._rating_label = rating_label

    # TODO HOMEWORK 2: FINISH IMPLEMENTATION
    # Need override because of rating label
    def compute_profile_length(self, u):
        """
        Computes the length of a user's profile vector.
        :param u: user
        :return: length
        """
    # TODO HOMEWORK 2: IMPLEMENT
    # Need override because of rating label.
        
        profile_rating = self.get_profile(u)[self._rating_label]
        ratings_square = profile_rating**2
        return ratings_square.sum()**0.5
    
    def compute_profile_means(self):

    # TODO HOMEWORK 2: IMPLEMENT
    # Need override because of shrinkage calculation
        for user in self.get_users():
            if user not in self._profile_means:
                mean_user_rating = self.get_profile(user)[self._rating_label].mean()
                self._profile_means[user] = mean_user_rating
                
    def compute_similarity_cache(self):

    # TODO HOMEWORK 2: IMPLEMENT
    # Need override because of user and item labels for indexing on columns
        count = 0
        for u in self.get_users():
            # TODO Rest of code here
            for v in self.get_users():
                if u != v:
                    sim = self.cosine(u, v)
                    num_of_overlaps = len(self.get_overlap(u,v))
                    
                    # Overlap Check
                    if num_of_overlaps == 0:
                        weighing_factor = 1
                    else:
                        weighing_factor = num_of_overlaps/(num_of_overlaps+self._shrinkage)
                    
                    sim *= weighing_factor
                    
                    if u not in self._sim_cache:
                        self._sim_cache[u] = {v:sim}
                    else:
                        self._sim_cache[u][v] = sim
                    
                    
            if count % 10 == 0:
                print ("Processed user {} ({})".format(u, count))
            count += 1
            
    def fit(self, ratings):
        User_KNN._ratings = ratings.set_index([self._user_label, self._item_label])
        self.compute_profile_lengths()
        self.compute_profile_means()
        self.compute_similarity_cache()

