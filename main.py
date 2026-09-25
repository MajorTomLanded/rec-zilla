import argparse
from model.data_client import DataClient
from model.recommender import Recommender
from model.cold_start import ColdStartRecommender

def main():
    parser = argparse.ArgumentParser(description="Get RecZilla Movie Recommendations for new and existing users.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recommend_parser = subparsers.add_parser("recommend", help="Recommend for an existing user.")
    recommend_parser.add_argument("user_id", help="e.g. course_2")
    recommend_parser.add_argument("-n", "--num-recommendations", type=int, default=5)

    cold_start_parser = subparsers.add_parser("cold-start", help="Recommend for a new user via free text.")
    cold_start_parser.add_argument("description", help="Free text describing what the user likes/dislikes")
    cold_start_parser.add_argument("-n", "--num-recommendations", type=int, default=5)
    cold_start_parser.add_argument("-k", "--neighbors", type=int, default=5)
    cold_start_parser.add_argument(
        "--cached-only", action="store_true",
        help="Skip fetching missing warm-user embeddings; use only what's already cached (useful when rate limited)."
    )

    subparsers.add_parser("warm-up", help="Pre-compute and cache embeddings for all active users.")

    args = parser.parse_args()
    data_client = DataClient()
    recommender = Recommender(data_client)

    # only construct the cold start recommender for commands that need it
    cold_start = None
    if args.command in ("warm-up", "cold-start"):
        cold_start = ColdStartRecommender(data_client)
        cold_start.warm_up(fetch_missing=not getattr(args, "cached_only", False))

    if args.command == "recommend":
        print(recommender.recommend(args.user_id, n=args.num_recommendations))
    elif args.command == "cold-start":
        new_user_id = cold_start.seed_new_user(args.description, k=args.neighbors)
        print(recommender.recommend(new_user_id, n=args.num_recommendations))

if __name__ == "__main__":
    main()