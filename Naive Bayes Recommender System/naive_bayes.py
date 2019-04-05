# Homework 4
# INFO 4871/5871, Spring 2019
# Robin Burke
# University of Colorado, Boulder

from lenskit.algorithms import Recommender
from lenskit.algorithms.basic import UnratedItemCandidateSelector

import numpy as np
import pandas as pd


class NaiveBayesRecommender(Recommender):

    _count_tables = {}
    _item_features = None
    _nb_table = None
    _min_float = np.power(2.0, -149)

    def __init__(self, item_features=None, thresh=2.9, alpha=0.01, beta=0.01):
        self._count_tables = {}
        self._item_features = item_features
        self.selector = UnratedItemCandidateSelector()
        self._nb_table = NaiveBayesTable(thresh, alpha, beta)
        self.ensure_minimum_score(alpha)
        self.ensure_minimum_score(beta)
        
    # TODO: HOMEWORK 4
    def fit(self, ratings, *args, **kwargs):
        # Must fit the selector
        self.selector.fit(ratings)

        self._nb_table.reset()
        # For each rating
            # Get associated item features
            # Update NBTable
        for index, row in ratings.iterrows():
            user, rating, item = row['user'], row['rating'], row['item']
            
            features = self.get_features_list(item)
            self._nb_table.process_rating(user, rating, features)

    # TODO: HOMEWORK 4
    # Should return ordered data frame with items and score
    def recommend(self, user, n=None, candidates=None, ratings=None):
        # n is None or zero, return DataFrame with an empty item column
        if n is None or n == 0:
            return pd.DataFrame({'item': []})

        if candidates is None:
            candidates = self.selector.candidates(user, ratings)

        # Initialize scores
        scores = []
        # for each candidate
        for candidate in candidates:
            # Score the candidate for the user
            score = self.score_item(user, candidate)
            # Build list of candidate, score pairs
            lists = [candidate, score]
            scores.append(lists)
        # Turn result into data frame
        scores = pd.DataFrame(scores, columns=['item', 'score'])
        # Retain n largest scoring rows (nlargest)
        scores = scores.nlargest(n, 'score')
        # Sort by score (sort_values)
        scores = scores.sort_values(by = 'score', ascending = False)
        # return data frame
        return scores

    # TODO: HOMEWORK 4
    # Helper function to return a list of features for an item from features data frame
    def get_features_list(self, item):
        if item not in self._count_tables:
            self._count_tables[item] = self._item_features[self._item_features.item == item]['feature']
        return self._count_tables[item]

    # TODO: HOMEWORK 4
    def score_item(self, user, item):
        # get the features
        # initialize the liked and nliked scores with the base probability
        features = self.get_features_list(item)
        liked_scores = self._nb_table.user_prob(user, True)
        nliked_scores = self._nb_table.user_prob(user, False)
        # for each feature
            # update scores by multiplying with conditional probability
        for feature in features:
            liked_scores *= self._nb_table.user_feature_prob(user, feature, True)
            
            nliked_scores *= self._nb_table.user_feature_prob(user, feature, False)
        # Handle the case when scores go to zero.
        liked_scores = self.ensure_minimum_score(liked_scores)
        nliked_scores = self.ensure_minimum_score(nliked_scores)
        # Compute log-likelihood
        log_likelihood = np.log(liked_scores) - np.log(nliked_scores)
        # Handle zero again
        log_likelihood = self.ensure_minimum_score(log_likelihood)
        # Return result
        return log_likelihood

    # DO NOT ALTER
    def get_params(self, deep=True):

        return {'item_features': self._item_features,
                'thresh': self._nb_table.thresh,
                'alpha': self._nb_table.alpha,
                'beta': self._nb_table.beta}

    # DO NOT ALTER
    def ensure_minimum_score(self, val):
        if val == 0.0:
            return self._min_float
        else:
            return val


# TODO: HOMEWORK 4
# Helper class
class NaiveBayesTable:
    liked_cond_table = {}
    nliked_cond_table = {}
    liked_table = {}
    nliked_table = {}
    thresh = 0
    alpha = 0.01
    beta = 0.01

    # TODO: HOMEWORK 4
    def __init__(self, thresh=2.9, alpha=0.01, beta=0.01):
        self.thresh = thresh
        self.alpha = alpha
        self.beta = beta
    # TODO: HOMEWORK 4
    # Reset all the tables
    def reset(self):
        self.liked_cond_table = {}
        self.nliked_cond_table = {}
        self.liked_table = {}
        self.nliked_table = {}
    # TODO: HOMEWORK 4
    # Return the count for a feature for a user (either liked or ~liked)
    # Should be robust if the user or the feature are not currently in table: return 0 in these cases
    def user_feature_count(self, user, feature, liked=True):
        if liked:
            if user not in self.liked_cond_table:
                return 0
            elif feature not in self.liked_cond_table[user]:
                return 0
            return self.liked_cond_table[user][feature]
        else:
            if user not in self.nliked_cond_table:
                return 0
            elif feature not in self.nliked_cond_table[user]:
                return 0
            return self.nliked_cond_table[user][feature]

    # TODO: HOMEWORK 4
    # Sets the count for a feature for a user (either liked or ~liked)
    # Should be robust if the user or the feature are not currently in table. Create appropriate entry or entries
    def set_user_feature_count(self, user, feature, count, liked=True):
        if liked:
            if user not in self.liked_cond_table:
                self.liked_cond_table[user] = {}
            if feature not in self.liked_cond_table[user]:
                self.liked_cond_table[user][feature] = 0
            self.liked_cond_table[user][feature] = count
        else:
            if user not in self.nliked_cond_table:
                self.nliked_cond_table[user] = {}
            if feature not in self.nliked_cond_table[user]:
                self.nliked_cond_table[user][feature] = 0
            self.nliked_cond_table[user][feature] = count

    def incr_user_feature_count(self, user, feature, liked=True):
        val = self.user_feature_count(user, feature, liked)
        self.set_user_feature_count(user, feature, val+1, liked)

    # TODO: HOMEWORK 4
    # Computes P(f|L) or P(f|~L) as the observed ratio of features and total likes/dislikes
    # Should include smooting with beta value
    def user_feature_prob(self, user, feature, liked=True):
        likelihood = (self.user_feature_count(user,feature,liked) + self.beta)/(self.user_count(user,liked) + 2*self.beta)
        return likelihood

    # TODO: HOMEWORK 4
    # Return the liked (disiked) count for a user (
    # Should be robust if the user is not currently in table: return 0 in this cases
    def user_count(self, user, liked=True):
        if liked:
            if user in self.liked_table:
                return self.liked_table[user]
        else:
            if user in self.nliked_table:
                return self.nliked_table[user]
        return 0

    # TODO: HOMEWORK 4
    # Sets the liked/disliked count for a user
    # Should be robust if the user is not currently in table. Create appropriate entry
    def set_user_count(self, user, value, liked=True):
        if liked:
            if user not in self.liked_table:
                self.liked_table[user] = 0
            self.liked_table[user] = value
        else:
            if user not in self.nliked_table:
                self.nliked_table[user] = 0
            self.nliked_table[user] = value

    def incr_user_count(self, user, liked=True):
        val = self.user_count(user, liked)
        self.set_user_count(user, val+1, liked)

    # TODO: HOMEWORK 4
    # Computes P(L) or P(~L) as the observed ratio of liked/dislike and total rated item count
    # Should include smooting with alpha value
    def user_prob(self, user, liked=True):
        prior = (self.user_count(user,liked) + self.alpha)/(self.user_count(user, True) + self.user_count(user, False) + 2*self.alpha)
        
        return prior

    # TODO:HOMEWORK 4
    # Update the table to take into account one new rating
    def process_rating(self, user, rating, features):

        # Determine if liked or disliked
        if rating < self.thresh:
            liked = False
        else:
            liked = True
        # Increment appropriate count for the user
        self.incr_user_count(user, liked)
        # For each feature
            # Increment appropriate feature count for the user

        for feature in features:
            self.incr_user_feature_count(user, feature, liked)