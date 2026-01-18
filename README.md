# Collaborative Filtering Recommendation System

A Python implementation of collaborative filtering for building recommendation systems using matrix factorization and gradient descent optimization.

## 📋 Overview

This project implements a collaborative filtering algorithm that learns latent features of users and items to predict user preferences. The model is trained using gradient descent with custom line search optimization and includes comprehensive gradient checking for verification.

## 🚀 Features

- **Matrix Factorization**: Breaks down user-item rating matrix into user and item feature matrices
- **Regularized Cost Function**: Prevents overfitting with L2 regularization
- **Gradient Descent Optimization**: Custom implementation with adaptive learning rate
- **Line Search**: Backtracking line search for optimal step size selection
- **Gradient Checking**: Numerical verification of gradient calculations
- **Progress Tracking**: Real-time cost monitoring during training

## 📦 Installation

```bash
git clone https://github.com/ChethanPutran/Collaborative-Filtering-for-a-recommendation-system
cd Collaborative-Filtering-for-a-recommendation-system
pip install -r requirements.txt
```

## 🏗️ Architecture

The system consists of four main components:

1. **Cost Function** (`cofi_cost_func`): Computes collaborative filtering cost and gradients
2. **Optimizer** (`minimize_cost`): Gradient descent with convergence checking
3. **Line Search** (`line_search`): Adaptive learning rate optimization
4. **Gradient Checker** (`check_cost_function`): Numerical gradient verification

## 🧮 Mathematical Formulation

The model predicts ratings using:
```
Ŷ = X × Θᵀ
```
Where:
- `X` = Item feature matrix (n_items × n_features)
- `Θ` = User preference matrix (n_users × n_features)

Cost function with regularization:
```
J(X,Θ) = ½∑(Ŷ - Y)² + (λ/2)(∑||X||² + ∑||Θ||²)
```


### Making Predictions

```bash

# Usage examples:

# Show help
python main.py --help

# Just train the model
python main.py --train

# Just predict for a user
python main.py --predict 42

# Predict for user 42 with 10 recommendations
python main.py --predict 42 --n_recommendations 10

# Train and predict (default user 1)
python main.py --train --predict 1

# Train and predict for specific user
python main.py --train --user_id 15

# Run default pipeline (train + predict for user 1)
python main.py
```

## 🎯 Applications

- Movie recommendation systems
- E-commerce product recommendations
- Music streaming services
- Content personalization platforms

## 📝 Key Concepts

1. **Collaborative Filtering**: Predicts user preferences based on similar users' behavior
2. **Latent Features**: Hidden characteristics learned from user-item interactions
3. **Regularization**: Prevents overfitting to sparse rating data
4. **Matrix Factorization**: Decomposes large rating matrix into smaller feature matrices

