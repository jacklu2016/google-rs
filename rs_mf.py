#the code was rewrite with tensorflow2.0
#tensorflow implementation of collaborate filter recommendation system
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
tf.compat.v1.disable_eager_execution()
#tf.disable_v2_behavior()
import altair as alt
#alt.renderers.enable('vegascope')
import gspread


#tf.logging.set_verbosity(tf.logging.ERROR)

# Add some convenience functions to Pandas DataFrame.
pd.options.display.max_rows = 10
pd.set_option('display.max_columns', None)
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

ratting_cols = ['user_id','movie_id','rating','unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data',sep='\t',names=ratting_cols,encoding='latin-1')

#the movie file contains a binary feature for each genre
genre_cols = ["genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
movie_cols = ['movie_id','title','release_date','video_release_date','imba_url'] + genre_cols

movies = pd.read_csv('ml-100k/u.item',sep='|',names=movie_cols,encoding='latin-1')

print(movies.head())
#Since the ids start at 1, we shift them to start at 0. 数据集的ids都会从1开始，改为从0开始
users['user_id'] = users['user_id'].apply(lambda x: x - 1)
movies['movie_id'] = movies['movie_id'].apply(lambda x: x - 1)
movies['year'] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
ratings['movie_id'] = ratings['movie_id'].apply(lambda x: x - 1)
ratings['user_id'] = ratings['user_id'].apply(lambda x: x - 1)
ratings['rating'] = ratings['rating'].apply(lambda x: x - 1)

#Compute the number of movies to which a genre is assigned.
#计算每种电影分类中有多少部电影
genre_occurences = movies[genre_cols].sum().to_dict()

print(genre_occurences)
# Since some movies can belong to more than one genre, we create different
# 'genre' columns as follows:
# - all_genres: all the active genres of the movie.
# - genre: randomly sampled from the active genres.
#由于一些电影属于多个分类，我们创建2个不同的分类字段，all_genres 电影的所有分类 ，genre,从所有分类中随机取一个分类
def mark_genre(movies,genres):
    def sample_genre(gs):
        #gs是movie的所有分类值的集合，例如id为1的movie的gs值为
        # (0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        active = [genre for genre,g in zip(genres,gs) if g == 1]
        if len(active) == 0:
            active = 'Other'
        return np.random.choice(active)

    def all_genre(gs):
        active = [genre for genre,g in zip(genres,gs) if g == 1]
        if len(active) == 0:
            active = 'Other'
        return '-'.join(active)

    movies['genre'] = [sample_genre(gs) for gs in zip(*[movies[genre] for genre in genres])]
    movies['all_genres'] = [all_genre(gs) for gs in zip(*[movies[genre] for genre in genres])]

mark_genre(movies,genre_cols)

# Create one merged DataFrame containing all the movielens data.
#把movies，users，ratings合并为一个数据框
movielens = ratings.merge(movies,on='movie_id').merge(users,on='user_id')

def split_dataframe(df,holdhout_fraction=0.1):
    test = df.sample(frac=holdhout_fraction,replace=False)
    train = df[~df.index.isin(test.index)]
    return train,test

#Before we dive into model building, let's inspect our MovieLens dataset. It is usually helpful to understand the statistics of the dataset.

#Users
#We start by printing some basic statistics describing the numeric user features.
# @title Altair visualization code (run this cell)
# The following functions are used to generate interactive Altair charts.
# We will display histograms of the data, sliced by a given attribute.

occupation_filter = alt.selection_multi(fields=["occupation"])
occupation_chart = alt.Chart().mark_bar().encode(
    x="count()",
    y=alt.Y("occupation:N"),
    color=alt.condition(
        occupation_filter,
        alt.Color("occupation:N", scale=alt.Scale(scheme='category20')),
        alt.value("lightgray")),
).properties(width=300, height=300, selection=occupation_filter)

def filtered_hist(field, label, filter):
  """Creates a layered chart of histograms.
  The first layer (light gray) contains the histogram of the full data, and the
  second contains the histogram of the filtered data.
  Args:
    field: the field for which to generate the histogram.
    label: String label of the histogram.
    filter: an alt.Selection object to be used to filter the data.
  """
  base = alt.Chart().mark_bar().encode(
      x=alt.X(field, bin=alt.Bin(maxbins=10), title=label),
      y="count()",
  ).properties(
      width=300,
  )
  return alt.layer(
      base.transform_filter(filter),
      base.encode(color=alt.value('lightgray'), opacity=alt.value(.7)),
  ).resolve_scale(y='independent')

users_ratings = (
    ratings
    .groupby('user_id', as_index=False)
    .agg({'rating': ['count', 'mean']})
    .flatten_cols()
    .merge(users, on='user_id')
)
#users_ratings_occup = users_ratings.groupby('occupation',as_index=False).agg({'rating count': ['count']})
print(users_ratings[(users_ratings['occupation'].isin(['administrator']))])

# Create a chart for the count, and one for the mean.
alt.hconcat(
    filtered_hist('rating count', '# ratings / user', occupation_filter),
    filtered_hist('rating mean', 'mean user rating', occupation_filter),
    occupation_chart,
    data=users_ratings)

movies_ratings = movies.merge(
    ratings
    .groupby('movie_id', as_index=False)
    .agg({'rating': ['count', 'mean']})
    .flatten_cols(),
    on='movie_id')

genre_filter = alt.selection_multi(fields=['genre'])
genre_chart = alt.Chart().mark_bar().encode(
    x="count()",
    y=alt.Y('genre'),
    color=alt.condition(
        genre_filter,
        alt.Color("genre:N"),
        alt.value('lightgray'))
).properties(height=300, selection=genre_filter)

alt.hconcat(
    filtered_hist('rating count', '# ratings / movie', genre_filter),
    filtered_hist('rating mean', 'mean movie rating', genre_filter),
    genre_chart,
    data=movies_ratings)

'''The rating matrix could be very large and, in general, most of the entries are unobserved,
 since a given user will only rate a small subset of movies. For effcient representation, 
 we will use a tf.SparseTensor. A SparseTensor uses three tensors to represent the matrix: 
 tf.SparseTensor(indices, values, dense_shape) represents a tensor,
  where a value  Aij=a  is encoded by setting indices[k] = [i, j] and values[k] = a.
   The last tensor dense_shape is used to specify the shape of the full underlying matrix.
user_id	movie_id	rating
0	0	5.0
0	1	3.0
1	3	1.0
SparseTensor(
  indices=[[0, 0], [0, 1], [1,3]],
  values=[5.0, 3.0, 1.0],
  dense_shape=[2, 4])
'''

def build_rating_sparse_tensor(rating_df):
    indices = rating_df[['user_id','movie_id']].values
    values = rating_df['rating'].values
    dense_shape = [users.shape[0],movies.shape[0]]
    return tf.sparse.SparseTensor(indices=indices,values=values,dense_shape=dense_shape)


def sparse_mean_square_error(sparse_rating,user_embedings,movie_embedings):
    prediction = tf.gather_nd(
        tf.matmul(user_embedings,movie_embedings,transpose_b=True),
        indices=sparse_rating.indices
    )
    loss = tf.losses.mean_absolute_error(sparse_rating.values,prediction)
    return loss

'''
Note: One approach is to compute the full prediction matrix  UV⊤ , 
then gather the entries corresponding to the observed pairs. The memory cost of this approach is  O(NM) . 
For the MovieLens dataset, this is fine, as the dense  N×M  matrix is small enough to fit
 in memory ( N=943 ,  M=1682 ).

Another approach (given in the alternate solution below) is to only gather the embeddings of the observed pairs,
 then compute their dot products. The memory cost is  O(|Ω|d)  where  d  is the embedding dimension. 
 In our case,  |Ω|=105 , and the embedding dimension is on the order of  10 , 
 so the memory cost of both methods is comparable. But when the number of users or movies is much larger, 
 the first approach becomes infeasible.
'''
def sparse_mean_square_error(sparse_rating,user_embedings,movie_embedings):
    prediction = tf.reduce_sum(
        tf.gather(user_embedings,sparse_rating.indices[:,0]) *
        tf.gather(movie_embedings,sparse_rating.indices[:,1]),axis=1
    )
    loss = tf.losses.mean_absolute_error(sparse_rating.values,prediction)
    return loss

class CFModel():
    def __init__(self,embedding_vars,loss,metrics=None):
        self._embedding_vars = embedding_vars
        self._loss = loss
        self._metrics = metrics
        self._embeddings = {k:None for k in embedding_vars}
        self._session = None

    @property
    def embeddings(self):
        return self._embeddings

    def train(self,num_iterations=100,learning_rate=1.0,plot_results=True,
              optimizer=tf.compat.v1.train.GradientDescentOptimizer):
        with self._loss.graph.as_default():
            opt = optimizer(learning_rate)
            train_op = opt.minimize(self._loss)
            local_init_op = tf.group(tf.compat.v1.variables_initializer(opt.variables()),
                                     tf.compat.v1.local_variables_initializer())
            if self._session is None:
                self._session = tf.compat.v1.Session()
                with self._session.as_default():
                    self._session.run(tf.compat.v1.global_variables_initializer())
                    self._session.run(tf.compat.v1.tables_initializer())
                    tf.compat.v1.train.start_queue_runners()

            with self._session.as_default():
                local_init_op.run()
                iterations = []
                metrics = self._metrics or ({},)
                metrics_vals = [collections.defaultdict(list) for _  in self._metrics]

                #train and append result
                for i in range(num_iterations + 1):
                    _,results = self._session.run((train_op,metrics))
                    if (i % 10 == 0) or i == num_iterations:
                        print('\r iteration %d:' % i + ','.join(
                            ['%s=%f' % (k,v) for r in results for k,v in r.items()]),end='')
                        iterations.append(i)

                        for metrics_val,result in zip(metrics_vals,results):
                            for k,v in result.items():
                                metrics_val[k].append(v)

                for k, v in self._embedding_vars.items():
                    self._embeddings[k] = v.eval()

                if plot_results:
                    num_subplots = len(metrics) + 1
                    fig = plt.figure()
                    fig.set_size_inches(num_subplots * 10,8)
                    for i ,metric_vals in enumerate(metrics_vals):
                        ax = fig.add_subplot(1,num_subplots,i + 1)
                        for k, v in  metric_vals.items():
                            ax.plot(iterations,v,label=k)
                        ax.set_xlim([1,num_iterations])
                        ax.legend()
                return results

def build_model(ratings,embedding_dim=3,init_stddev=1.):
    #split the dataset into train and test
    train,test = split_dataframe(ratings)

    #sparse tensor representation of the train and test
    a_train = build_rating_sparse_tensor(train)
    a_test = build_rating_sparse_tensor(test)
    print(a_train.dense_shape[0],a_train.dense_shape[1])
    #initialize the variables using a normal distribution
    U = tf.Variable(tf.random.normal([a_train.dense_shape[0],embedding_dim],stddev=init_stddev))
    V = tf.Variable(tf.random.normal([a_train.dense_shape[1],embedding_dim],stddev=init_stddev))

    train_loss = sparse_mean_square_error(a_train,U,V)
    test_loss = sparse_mean_square_error(a_test,U,V)

    metrics = {'trainning_error':train_loss,'test_error':test_loss}
    embeddings = {'user_id':U,'movie_id':V}
    return CFModel(embeddings,train_loss,[metrics])

model = build_model(ratings,embedding_dim=30,init_stddev=0.5)
model.train(num_iterations=100,learning_rate=10.)

DOT = 'dot'
COSIONE = 'cosine'
def compute_score(query_embedding,item_embeddings,measure=DOT):
    """
    Computes the scores of the candidates given a query.
  Args:
    query_embedding: a vector of shape [k], representing the query embedding.
    item_embeddings: a matrix of shape [N, k], such that row i is the embedding
      of item i.
    measure: a string specifying the similarity measure to be used. Can be
      either DOT or COSINE.
  Returns:
    scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    u = query_embedding
    V = item_embeddings
    if measure == COSIONE:
        u = u / np.linalg.norm(u)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)

    scores = u.dot(V.T)
    return scores

#Equipped with this function, we can compute recommendations,
# where the query embedding can be either a user embedding or a movie embedding.
def user_recommendation(user_id,model,measure=DOT,exclude_rated=False,k=6):
    scores = compute_score(model.embeddings['user_id'][user_id],model.embeddings['movie_id'])
    score_key = 'score ' + measure
    df = pd.DataFrame({score_key: list(scores),
                       'movie_id': movies['movie_id'],
                       'title': movies['title'],
                       'genres': movies['all_genres'],})
    if exclude_rated:
        rated_movie_ids = ratings[ratings['user_id'] == user_id]['movie_id'].values
        df = df[df.movie_id.apply(lambda x: x not in rated_movie_ids)]

    display.display(df.sort_values(score_key,ascending=False).head(k))


def movie_neighbors(movie_title,model,meansure=DOT,k=6):
    query_movie_ids = movies[movies['title'].str.contains(movie_title)].index.values
    titles = movies.iloc[query_movie_ids]['title'].values
    if len(query_movie_ids) == 0:
        raise ValueError('Found no moive with title:%s' % movie_title)
    print('Nearest neighbors of :%s. ' % titles[0])
    if len(titles) > 1:
        print('[Found more than one matching movies,other candidates {} '.format(",".join(titles[1:])))

    movie_id = query_movie_ids[0]
    score_key = 'socre ' + DOT
    scores = compute_score(model.embeddings['movie_id'][movie_id],model.embeddings['movie_id'])
    df = pd.DataFrame({score_key: scores,
                       'title': movies['title'],
                       'genres': movies['all_genres']})

    display.display(df.sort_values(score_key, ascending=False).head(k))


user_recommendation(940, model, exclude_rated=True)
movie_neighbors("Aladdin", model, DOT)
movie_neighbors("Aladdin", model, COSIONE)
