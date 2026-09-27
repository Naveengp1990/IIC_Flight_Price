"""
Run this script ONCE after placing train.csv in the project folder.

Command:
    python train_model.py

It creates:
    process.pkl
    xgb_model.pkl
"""

from model_pipeline import train_and_save


if __name__ == "__main__":
    train_and_save()
