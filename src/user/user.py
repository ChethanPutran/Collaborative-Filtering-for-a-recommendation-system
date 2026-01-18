import numpy as np

def get_new_user_ratings(movies):
    n = len(movies)

    ratings = np.zeros((n,1))
    
    # Entering ratings for a new user
    ratings[0] = 4  # Toy Story (1995)
    ratings[97] = 2 # Silence of the Lambs, The (1991)
    ratings[6] = 3  # Four Rooms (1995)
    ratings[11]= 5  # Get Shorty (1995)
    ratings[53]= 4  # Seven (Se7en) (1995)
    ratings[63]= 5  # Usual Suspects, The (1995)
    ratings[65]= 3  # Mighty Aphrodite (1995)
    ratings[68]= 3  # Postman, The (1997)
    ratings[182]= 4 # Liar Liar (1997)
    ratings[225]= 5 # Heat (1995)
    ratings[354]= 5 # Braveheart (1995)

    return ratings