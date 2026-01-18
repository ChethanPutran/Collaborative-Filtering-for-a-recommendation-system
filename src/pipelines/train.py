from src.data.data import load_movie_data, load_movie_list, normalize_ratings
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from src.models.collaborative_filtering import CollaborativeFiltering
from src.user.user import get_new_user_ratings


def train(use_trained=False, _lambda=10, alpha=0.001, num_iters=100, plot_cost=True):

    # Load Data
    X, Theta, Y, R, num_users, num_movies, num_features = load_movie_data()


    plt.imshow(Y, extent=[1, num_users, 1, num_movies])
    plt.ylabel('Movies')
    plt.xlabel('Users')
    plt.show()

    k_movies = 100
    k_users = 100

    plt.imshow(Y[:k_movies,:k_users], extent=[1, k_users, 1, k_movies])
    plt.ylabel('Movies')
    plt.xlabel('Users')
    plt.show()


    if use_trained:
        data = sio.loadmat("../data/weights.mat")
        X_i = data['X'].ravel()
        Theta_i = data['Theta'].ravel()
    else:
        # Set Initial Parameters (Theta, X)
        X_i = np.random.random((num_movies,num_features))
        Theta_i = np.random.random((num_users,num_features))

    # Let's rate a few movies
    my_ratings = get_new_user_ratings(load_movie_list())

    Theta_inp = np.concat([Theta_i,np.random.random((1,num_features))],axis=0)

    ### ==================  Learning Movie Ratings ====================
    # Prepare data matrix
    Y_new = np.column_stack((my_ratings,Y))
    R_new = np.column_stack(((my_ratings != 0).astype(np.uint8),R))
    Y_mean,Y_norm = normalize_ratings(Y_new,R_new)
    num_movies,num_users = Y_norm.shape

    print("No. of users :",num_users)
    print("No. of movies :",num_movies)
    print("No. of features :",num_features)


    collaborative_filtering = CollaborativeFiltering()
    
    min_cost,history= collaborative_filtering.train(X_i,Theta_inp,Y_norm, R_new,_lambda, alpha, num_iters)
    print("Minimum cost after training :",min_cost) 
   

    if plot_cost:
        plt.plot(history)
        plt.xlabel("No. iteration")
        plt.ylabel("Cost")
        plt.title(" Training cost vs No. iterations")

