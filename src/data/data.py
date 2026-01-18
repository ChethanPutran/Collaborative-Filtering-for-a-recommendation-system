import  numpy as np
import scipy.io as sio


def load_movie_data(movie_file='data/movies.mat', params_file='data/movieParams.mat'):
    """
    Load movie rating data and movie features from .mat files.
    Returns:
        X: num_movies x num_features matrix of movie features
        Theta: num_users x num_features matrix of user features
        Y: num_movies x num_users matrix of user ratings of movies
        R: num_movies x num_users matrix, where R(i, j) = 1 if the i-th movie was rated by the j-th user
        num_users: number of users
        num_movies: number of movies
        num_features: number of features
    """
    # Load data
    data1 = sio.loadmat(movie_file)
    data2 = sio.loadmat(params_file)

    Y,R = data1['Y'],data1['R']
    X = data2['X']
    Theta = data2['Theta']
    num_users = data2['num_users'].item()
    num_movies =  data2['num_movies'].item()
    num_features =  data2['num_features'].item()

    return X, Theta, Y, R, num_users, num_movies, num_features


def load_movie_list(movie_list_file='data/movie_ids.txt'):
    """ 
    Reads the fixed movie list in movie.txt and returns a cell array of the words
    """
    with open(movie_list_file) as f:
        movies_str = f.readlines()

    movies = {}
        
    for line in movies_str:
        idx = line.find(" ")
        movie_id = int(line[:idx])
        movie_name = line[idx+1:].rstrip("\n")
        movies[movie_id] = movie_name
    return movies


def normalize_ratings(Y,R):
    """ Preprocess data by subtracting mean rating for every movie (every row) so that each movie 
    has a rating of 0 on average, and returns the mean rating in Ymean.
    """
    m,n = Y.shape
    Y_mean = np.zeros((m, 1))
    Y_norm = np.zeros((m,n))
    for i in range(m):
        idx = np.where(R[i,:] == 1)
        Y_mean[i] = Y[i, idx].mean()
        Y_norm[i, idx] = (Y[i, idx] - Y_mean[i])
    return Y_mean,Y_norm
