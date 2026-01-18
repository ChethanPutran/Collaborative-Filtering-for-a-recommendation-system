from src.pipelines.train import train
from src.pipelines.inference import predict


if __name__ == "__main__":
    train()
    predict(1)  # Predict for user with ID 1