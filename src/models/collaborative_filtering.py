import numpy as np
import sys



class CollaborativeFiltering:
    def __init__(self):
        self.X = None
        self.Theta = None
        try:
            self.load_model()
        except:
            print('No pre-trained model found. Please train the model first.')
            # Optionally, you can raise an exception here
            # raise FileNotFoundError('Pre-trained model not found.')
            # Or initialize empty model parameters

    def predict(self, Y_mean):
        """
        Predict ratings for all users and movies

        Parameters:
        X_optim : array_like
            num_movies x num_features matrix of movie features
        Theta_optim : array_like
            num_users x num_features matrix of user features
        Y_mean : array_like
            num_movies x 1 vector of mean movie ratings
        """
        
        p = self.X @ self.Theta.T
        predictions = p[:,0] + Y_mean.ravel()
        predicted = np.argsort(predictions)[::-1]
        return predictions, predicted

    def predict_for_user(self,user_id):
        """
        Predict ratings for a specific user

        Parameters:
        user_id : int
            ID of the user for whom to make predictions 
        """
        theta = self.Theta[user_id,:]
        predictions = self.X @ theta.T
        predicted = np.argsort(predictions)[::-1]
        return predictions, predicted

    def collaborative_filtering_cost(self,
                                     X:np.ndarray,
                                     Theta:np.ndarray,
                                       Y:np.ndarray, R:np.ndarray, lmbda:float):
        """
        Collaborative Filtering Cost Function 

        Parameters:
        X : array_like
            num_movies x num_features matrix of movie features
        Theta : array_like
            num_users x num_features matrix of user features
        Y : array_like
            num_movies x num_users matrix of user ratings of movies
        R : array_like
            num_movies x num_users matrix, where R(i, j) = 1 if the i-th movie was rated by the j-th user
        lmbda : float
            Regularization parameter
        """
        J = 0
        X_grad = np.zeros(X.shape)
        Theta_grad = np.zeros(Theta.shape)
        
        errors = ( X @ Theta.T - Y ) * R
        regularization_theta = 0.5 * lmbda * np.sum(Theta**2)
        regularization_X = 0.5 * lmbda * np.sum(X**2)

        J = 0.5 * np.sum(errors**2) + regularization_theta + regularization_X

        # Gradient calculation
        # X_grad - num_movies x num_features matrix, containing the partial derivatives w.r.t. to each element of X
        X_grad[:,:] = errors @ Theta + lmbda * X

        #Theta_grad - num_users x num_features matrix, containing the partial derivatives w.r.t. to each element of Theta
        Theta_grad[:,:] = errors.T @ X + lmbda * Theta

        return J,X_grad, Theta_grad


    def train(self,X,Theta,Y, R,lmbda, alpha, num_iters,epsilon=0.0001):
        #Initializing Js
        previousJ=0
        currentJ=0
        history = np.zeros((num_iters,1))


        print('Training collaborative filtering...')
        for i in range(0,num_iters):
            # time.sleep(0.001)  # Adjust speed
            currentJ,grad_X, grad_Theta = self.collaborative_filtering_cost(X,Theta,Y,R,lmbda)
            alpha = self.line_search(X,Theta,grad_X,grad_Theta,Y,R,lmbda,alpha=alpha)
            history[i,:] = currentJ
            sys.stdout.write(f"\rIteration: {i+1}/{num_iters} Cost:{"{:.2f}".format(currentJ)}  ΔError: {"{:.2f}".format(abs(currentJ-previousJ))}")
            Theta[:] = Theta - alpha * grad_Theta
            X[:] = X - alpha * grad_X
            # print(e)
            #time.sleep(0.1)
            sys.stdout.flush()
            if(i>0):
                if(abs(currentJ-previousJ)<=epsilon):
                    print(f'Max no. of iterations({i}) reached!')
                    break
            previousJ = currentJ
        print('\nRecommender system learning completed.')
        self.X = X
        self.Theta = Theta
        self.save_model(X, Theta)
        return [previousJ, history]\

    
    def save_model(self, X, Theta, filepath="model/collaborative_filtering_model.npz"):
        """
        Save the trained model parameters to a file.

        Parameters:
        X : array_like
            num_movies x num_features matrix of movie features
        Theta : array_like
            num_users x num_features matrix of user features
        filepath : str
            Path to the file where the model will be saved
        """
        try:
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez(filepath, X=X, Theta=Theta)
            print(f'Model saved to {filepath}')
        except Exception as e:
            print(f'Error creating directory for model saving: {e}')
        
    
    def load_model(self, filepath="model/collaborative_filtering_model.npz"):
        """
        Load the trained model parameters from a file.

        Parameters:
        filepath : str
            Path to the file from which the model will be loaded
        """
        data = np.load(filepath)
        self.X = data['X']
        self.Theta = data['Theta']
        print(f'Model loaded from {filepath}')

    def line_search(self,X,Theta, d_X, d_Theta,Y,R,_lambda,max_steps=10,alpha=0.001,reduce_by=0.4):
        """
        Performs a simple line search to find the optimal step size.
       
        Parameters:
        X : array_like
            num_movies x num_features matrix of movie features
        Theta : array_like
            num_users x num_features matrix of user features
        d : array_like      
            Direction of the step.
        args : list
            Additional arguments to pass to the cost function.
        max_steps : int, optional
            Maximum number of step size reductions. Default is 10.
        alpha : float, optional
            Initial step size. Default is 0.001.
        reduce_by : float, optional
            Factor by which to reduce the step size. Default is 0.4.
        """
        best_alpha = alpha
        temp_X = np.zeros_like(X)
        temp_Theta = np.zeros_like(Theta)

        def get(alpha):
            temp_X[:] = X - alpha*d_X
            temp_Theta[:] = Theta - alpha*d_Theta
            return temp_X,temp_Theta
                
        best_cost, _ , _ = self.collaborative_filtering_cost(*get(best_alpha),Y,R,_lambda)
        
        for _ in range(max_steps):
            alpha = best_alpha * reduce_by  # Reduce step size
            new_cost,_,_ = self.collaborative_filtering_cost(*get(alpha),Y,R,_lambda)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_alpha = alpha
            else:
                break  # Stop if cost increases
        return best_alpha


    # def compute_numerical_gradient(self,J, theta,*args):
    #     """
    #     Computes the gradient using "finite differences" and returns a numerical estimate of the gradient
    #     """
        
    #     numgrad = np.zeros_like(theta)
    #     perturb = np.zeros_like(theta)

    #     n = theta.shape[0]
    #     e = 1e-4
    #     for p in range(n):
    #         # Set perturbation vector
    #         perturb[p] = e
    #         loss1 = J(theta - perturb,*args)[0]
    #         loss2 = J(theta + perturb,*args)[0]
            
    #         # Compute Numerical Gradient
    #         numgrad[p] = (loss2 - loss1) / (2*e)
    #         perturb[p] = 0

    #     return numgrad

    # def check_cost_function(self,lmbda=0):
    #     """ Checks cost function and gradients"""
    #     # Create small problem
    #     X_t = np.random.random((4, 3))
    #     Theta_t = np.random.random((5, 3))
        
    #     # Zip out most entries
    #     Y = X_t @ Theta_t.T
    #     Y[np.random.random(Y.shape) > 0.5] = 0
    #     R = np.zeros(Y.shape)
    #     R[Y != 0] = 1
        
    #     # Run Gradient Checking
    #     X = np.random.random((4, 3))
    #     Theta = np.random.random((5, 3))
    #     num_users = Y.shape[1]
    #     num_movies = Y.shape[0]
    #     num_features = Theta_t.shape[1] 

    #     theta_i = np.concatenate((X.ravel(),Theta.ravel()))
    #     numgrad = self.compute_numerical_gradient(self.collaborative_filtering_cost, X,R,*[Y,  R, lmbda, num_movies, num_features, num_users])
    #     [cost, grad] = self.collaborative_filtering_cost(theta_i,  Y, R, lmbda, num_movies, num_features, num_users)
        
    #     print([numgrad, grad])
        
    #     diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    #     print('Relative Difference: %g' % diff)