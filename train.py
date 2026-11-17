"""
Entry point to launch the Training Pipeline.
Run with: python train.py --help
"""
from src.train import parse_args, entrenar_pipeline

if __name__ == "__main__":
    args = parse_args()
    entrenar_pipeline(args)
