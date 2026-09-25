import time
from dotenv import load_dotenv
from google import genai
import pandas as pd
import uuid
from google.genai.errors import ClientError
from sklearn.metrics.pairwise import cosine_similarity
from model.data_client import DataClient

class ColdStartRecommender:
    llm_client: genai.Client
    data_client: DataClient
    active_users: pd.DataFrame
    active_user_embeddings: pd.DataFrame | None

    def __init__(self, data_client: DataClient) -> None:
        load_dotenv()
        self.llm_client = genai.Client()
        self.data_client = data_client
        self.active_users = self._build_active_users()
        self.active_user_embeddings = None

    def warm_up(self, fetch_missing: bool = True) -> pd.DataFrame:
        """
        Load (or compute and cache) embeddings for all active users.
        Call this once before find_similar_users/seed_new_user.
        Safe to call more than once - already-cached users are skipped.

        fetch_missing=False uses only whatever is already cached on disk
        and skips calling the embeddings API for missing users - useful
        when rate limited. The neighbor pool will just be smaller.
        """
        self.active_user_embeddings = self._load_active_user_embeddings(fetch_missing=fetch_missing)
        return self.active_user_embeddings

    def _ensure_warmed_up(self) -> None:
        if self.active_user_embeddings is None:
            raise RuntimeError("ColdStartRecommender.warm_up() must be called before this method.")

    def _build_active_users(self) -> pd.DataFrame:
        active_user_ids = set[str](self.data_client.ratings["user_id"].unique())
        active_users = self.data_client.users[self.data_client.users["user_id"].isin(active_user_ids)].copy()
        active_users["description_text"] = (
            active_users["self_description_likes"].fillna("") + " " +
            active_users["self_description_dislikes"].fillna("")
        ).str.strip()
        return active_users[active_users["description_text"] != ""]

    def _load_active_user_embeddings(self, fetch_missing: bool = True) -> pd.DataFrame:
        # if this ran already, you should have the vectors saved locally
        active_user_embeddings_path = self.data_client.DATA_DIR / "warm_embeddings.csv"
        all_ids = "course_" + self.active_users["user_id"].astype(str)

        if active_user_embeddings_path.exists():
            existing = pd.read_csv(active_user_embeddings_path, index_col=0)
        else:
            existing = pd.DataFrame()
        # saves happen after each batch (rate limiting problems)
        # check for remaining IDs; getting the embeddings from the LLM is idempotent
        remaining = self.active_users[~all_ids.isin(existing.index).values]

        if len(remaining) == 0:
            print('Embeddings loaded from disk. 0 missing user IDs.')
            return existing

        if not fetch_missing:
            print(f"Skipping fetch for {len(remaining)}/{len(all_ids)} missing users (cached-only mode). Using {len(existing)} cached embeddings.")
            return existing

        print(f"Missing  {len(remaining)}/{len(all_ids)} users. Getting content embeddings now....")
        texts = remaining["description_text"].tolist()
        ids = "course_" + remaining["user_id"].astype(str)
        total = len(texts)
        batch_size = 50

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids.iloc[i:i + batch_size]

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.llm_client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=batch_texts,
                    )
                    break
                except ClientError as e:
                    if e.code == 429:
                        if attempt == max_retries - 1:
                            raise RuntimeError(
                                f"Still rate limited after {max_retries} retries. "
                                f"{i}/{total} of this run's missing users are embedded and saved to disk - "
                                "rerun later to pick up where you left off, or pass fetch_missing=False "
                                "(--cached-only on the CLI) to proceed without the rest."
                            ) from e
                        print(f"Rate limited, waiting 60s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(60)
                    else:
                        raise

            # save batch to file
            batch_embeddings = [e.values for e in response.embeddings]
            batch_df = pd.DataFrame(batch_embeddings, index=batch_ids)
            batch_df.to_csv(active_user_embeddings_path, mode="a", header=not active_user_embeddings_path.exists())

            print(f"Embedded {min(i + batch_size, total)}/{total} remaining (saved)")
            time.sleep(2)

        return pd.read_csv(active_user_embeddings_path, index_col=0)

    def embed_text(self, text: str) -> list[float]:
        response = self.llm_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return response.embeddings[0].values

    def find_similar_users(self, cold_user_text: str, k: int = 5) -> pd.Series:
        self._ensure_warmed_up()
        cold_embedding = self.embed_text(cold_user_text)
        similarity_table = cosine_similarity([cold_embedding], self.active_user_embeddings.values)[0]
        neighbors = pd.Series(similarity_table, index=self.active_user_embeddings.index)
        return neighbors.sort_values(ascending=False).head(k)

    def seed_new_user(self, cold_user_text: str, k: int = 5) -> str:
        neighbors = self.find_similar_users(cold_user_text, k=k)
        neighbor_ratings = self.data_client.ratings_matrix.loc[neighbors.index]
        pseudo_ratings = neighbor_ratings.mean(axis=0).dropna()

        new_user_id = f"cold_{uuid.uuid4().hex[:8]}"
        self.data_client.ratings_matrix.loc[new_user_id] = pseudo_ratings

        return new_user_id