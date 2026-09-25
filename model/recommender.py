from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from model.data_client import DataClient

class Recommender:
    data_client: DataClient
    item_similarity_dataframe: pd.DataFrame

    def __init__(self, data_client: DataClient) -> None:
        self.data_client = data_client
        self.item_similarity_dataframe = self.create_item_similarity_dataframe()

    def create_item_similarity_dataframe(self) -> pd.DataFrame:
        ratings_filled = self.data_client.ratings_matrix.fillna(0)
        item_similarity = cosine_similarity(ratings_filled.T)
        return pd.DataFrame(item_similarity, index=ratings_filled.columns, columns=ratings_filled.columns)

    def predict_rating(self, user_id: str, movie_id: str, neighborhood_size: int = 10) -> float | None:
        user_ratings = self.data_client.ratings_matrix.loc[user_id].dropna()
        candidates = self.item_similarity_dataframe.loc[movie_id, user_ratings.index]
        candidates = candidates[candidates > 0]
        neighborhood = candidates.sort_values(ascending=False).head(neighborhood_size)
        if neighborhood.sum() == 0:
            return None
        return (neighborhood * user_ratings[neighborhood.index]).sum() / neighborhood.sum()

    def recommend(self, user_id: str, n: int = 5) -> pd.Series:
        unrated = self.data_client.ratings_matrix.loc[user_id][self.data_client.ratings_matrix.loc[user_id].isna()].index
        candidates = {m: self.predict_rating(user_id, m) for m in unrated}
        candidates = {m: p for m, p in candidates.items() if p is not None}
        return pd.Series(candidates).sort_values(ascending=False).head(n)
