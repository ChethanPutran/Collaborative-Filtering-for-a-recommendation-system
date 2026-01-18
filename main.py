# main.py
import sys
import argparse
from src.pipelines.train import train
from src.pipelines.inference import predict

def main():
    parser = argparse.ArgumentParser(description="Recommendation System CLI")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=int, help="Predict for user ID")
    parser.add_argument("--n_recommendations", type=int, default=5, 
                       help="Number of recommendations to show")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    if args.train:
        print("Training the recommendation model...")
        train()
        print("Training completed successfully!")
    
    if args.predict is not None:
        print(f"Generating recommendations for user {args.predict}...")
        predict(args.predict, n_recommendations=args.n_recommendations)
    
    # Default behavior if no arguments provided (backward compatibility)
    if len(sys.argv) == 1:
        print("No arguments provided. Running default pipeline...")
        # train()
        predict(1, n_recommendations=5)

if __name__ == "__main__":
    main()