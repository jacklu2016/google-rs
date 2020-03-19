from __future__ import print_function

import numpy as np
import pandas as pd
import collections
from mpl_toolkits.mplot3d import Axes3D
from IPython import display
from matplotlib import pyplot as plt
import sklearn
import sklearn.manifold
import tensorflow as tf
import altair as alt
import gspread


tf.logging.set_verbosity(tf.logging.ERROR)

# Add some convenience functions to Pandas DataFrame.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format


def mask(df,key,function):
	"""Return a filtered dataframe ,by applying function to key"""
	return df[function(df[key])]


def flatten_cols(df):
	df.columns = [' '.join(col).strip() for col in df.columns.values]
	return df


pd.DataFrame.mask = mask
pd.DataFrame.flatten_cols = flatten_cols

#load each data
user_cols = ['user_id','age','sex','occupation','zip_code']
users = pd.read_csv('ml-100k/u.user',sep='|',names=user_cols,encoding='latin-1')

ratting_cols = ['user_id','movie_id','ratting','unix_timestamp']
rattings = pd.read_csv('ml-100k/u.data',sep='\t',names=ratting_cols,encoding='latin-1')

#the movie file contains a binary feature for each genre
genre_cols = ["genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
movie_cols = ['movie_id','title','release_date','video_release_date','imba_url'] + genre_cols

movies = pd.read_csv('ml-100k/u.item',sep='|',names=movie_cols,encoding='latin-1')

