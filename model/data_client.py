import urllib.request
import zipfile
from pathlib import Path
import pandas as pd


def _require_columns(df: pd.DataFrame, required: set[str], source: str) -> None:
    """Raise if a dataframe loaded from `source` is missing expected columns."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing expected columns: {sorted(missing)}")


class DataClient:
    DATA_DIR: Path
    movies: pd.DataFrame
    movielens_links: pd.DataFrame
    movielens_ratings: pd.DataFrame
    ratings: pd.DataFrame
    ratings_matrix: pd.DataFrame
    users: pd.DataFrame

    def __init__(self) -> None:
        self.DATA_DIR = Path("data")
        self.download_course_data()
        self.download_movielens_data()

        self.movies = pd.read_csv(self.DATA_DIR / "movies.csv.gz")
        _require_columns(self.movies, {"movie_id", "title", "genres", "tmdb_id"}, "movies.csv.gz")

        self.movielens_links = pd.read_csv(self.DATA_DIR / "ml-latest-small" / "links.csv")
        _require_columns(self.movielens_links, {"movieId", "tmdbId"}, "ml-latest-small/links.csv")

        self.movielens_ratings = pd.read_csv(self.DATA_DIR / "ml-latest-small" / "ratings.csv")
        _require_columns(self.movielens_ratings, {"userId", "movieId", "rating"}, "ml-latest-small/ratings.csv")

        self.ratings = pd.read_csv(self.DATA_DIR / "events.csv.gz")
        _require_columns(self.ratings, {"user_id", "movie_id", "rating"}, "events.csv.gz")

        self.users = pd.read_csv(self.DATA_DIR / "users.csv.gz")
        _require_columns(
            self.users,
            {"user_id", "self_description_likes", "self_description_dislikes"},
            "users.csv.gz",
        )

        self.ratings_matrix = self.make_combined_ratings_matrix()

    def download_file(self, url: str, destination: Path) -> None:
            """
            Download a file from a URL and save it to a destination path.
            """
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                urllib.request.urlretrieve(url, destination)


    def download_course_data(self) -> None:
        """
        Download the course data from mlip-cmu-online.
        """
        url: str = "https://github.com/mlip-cmu-online/public-data/raw/refs/heads/main/m0/data/"
        filenames: list[str] = [
            "events.csv.gz",
            "users.csv.gz",
            "movies.csv.gz"
        ]
        for filename in filenames:
            self.download_file(f"{url}/{filename}", self.DATA_DIR / filename)

    def download_movielens_data(self) -> None:
        """
        Download movielens data for additional interaction data
        """
        url: str = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zip_path: Path = self.DATA_DIR / "ml-latest-small.zip"
        ratings_path: Path = self.DATA_DIR / "ml-latest-small" / "ratings.csv"
        self.download_file(url, zip_path)
        if not ratings_path.exists():
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(self.DATA_DIR)


    def make_combined_ratings_matrix(self) -> pd.DataFrame:
        """
        Make a combined ratings matrix from the course and movielens data.
        """
        # merge course and movielens interactions on tmdb id
        ratings_join_table = self.movies.merge(self.movielens_links, left_on="tmdb_id", right_on="tmdbId", how="inner")[["movie_id", "movieId"]]

        movielens_ratings_mapped = self.movielens_ratings.merge(ratings_join_table, on="movieId", how="inner")[["userId", "movie_id", "rating"]]

        # namespace the users (no user merge)
        movielens_ratings_mapped["user_id"] = "movielens_" + movielens_ratings_mapped["userId"].astype(str)

        coursedata_ratings_namespaced = self.ratings[["user_id", "movie_id", "rating"]].copy()
        coursedata_ratings_namespaced["user_id"] = "course_" + coursedata_ratings_namespaced["user_id"].astype(str)

        # normalize the ratings using z-score
        course_rating_mean = coursedata_ratings_namespaced["rating"].mean()
        course_rating_std = coursedata_ratings_namespaced["rating"].std()

        coursedata_ratings_namespaced["rating_norm"] = (
            (coursedata_ratings_namespaced["rating"] - course_rating_mean) / course_rating_std
        )

        movielens_ratings_mean = movielens_ratings_mapped["rating"].mean()
        movielens_ratings_std = movielens_ratings_mapped["rating"].std()

        movielens_ratings_mapped["rating_norm"] = (
            (movielens_ratings_mapped["rating"] - movielens_ratings_mean) / movielens_ratings_std
        )

        # combine into one long table
        ratings_combined = pd.concat([
            coursedata_ratings_namespaced[["user_id", "movie_id", "rating_norm"]],
            movielens_ratings_mapped[["user_id", "movie_id", "rating_norm"]],
        ])
        # pivot
        ratings_matrix = ratings_combined.pivot_table(index="user_id", columns="movie_id", values="rating_norm")
        return ratings_matrix