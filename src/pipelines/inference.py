from src.data.data import load_movie_list
from src.models.collaborative_filtering import CollaborativeFiltering


def predict(user_id, get_top_n=10):
        print("\nTop recommendations for you:")
        movie_list = load_movie_list()
        collaborative_filtering = CollaborativeFiltering()
        predictions, predicted = collaborative_filtering.predict_for_user(user_id)
        for i in range(get_top_n):
            j = predicted[i]
            print("Predicting rating %.1f for movie %s" % (predictions[j], movie_list[j]))  