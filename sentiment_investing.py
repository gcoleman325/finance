# adapted from https://www.youtube.com/watch?v=9Y3yaoi9rUQ
import pandas as pd
import numpy as mp
import matplotlib.pyplot as plt
import datetime as dt
import kaggle
import os

twitter_data = pd.read_csv("C:/Users/ellie/OneDrive/Documents/GitHub/sentiment/training.1600000.processed.noemoticon.csv")
twitter_data.head()
