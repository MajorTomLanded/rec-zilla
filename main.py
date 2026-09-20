import argparse
from model.recommender import Recommender

def main():
    parser = argparse.ArgumentParser(description="Get RecZilla Movie Recommendations for a user.")
    parser.add_argument("user_id", help="e.g. course_2")
    parser.add_argument("-n", "--num-recommendations", type=int, default=5)
    args = parser.parse_args()

    recommender = Recommender()
    print(recommender.recommend(args.user_id, n=args.num_recommendations))

if __name__ == "__main__":
    main()