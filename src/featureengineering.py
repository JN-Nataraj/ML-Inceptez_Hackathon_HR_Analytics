from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineering (BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Example feature engineering: Create a new feature 'total_training_score'
        X = X.copy()
        X['new_joinee'] = X['previous_year_rating'].isna().astype(int)
        X['previous_year_rating'] = X['previous_year_rating'].fillna(0)
        return X