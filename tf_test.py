import tensorflow as tf
import pandas as pd
import numpy as np

columns_name = ['id','cat']
data = pd.DataFrame({'id':[0,1,2,3,4,5,6,7,8,9],
                     'cat':['test','train','eval','train','train','test','eval','eval','test','train'],
                     'dog':['test1','train1','eval1','train1','train1','test1','eval1','eval1','test1','train1'],
                     'mouse':['test2','train2','eval2','train2','train2','test2','eval2','eval2','test2','train2']})
features_new = {key: np.array(value) for key, value in dict(data).items()}
print(features_new)

cate_eb=tf.feature_column.categorical_column_with_vocabulary_list('cat',
                                                                  vocabulary_list=['test','train','eval'],
                                                                  default_value=1 )
cate_eb1=tf.feature_column.categorical_column_with_vocabulary_list('dog',
                                                                  vocabulary_list=['test1','train1','eval1'],
                                                                  default_value=1 )
cate_eb2=tf.feature_column.categorical_column_with_vocabulary_list('mouse',
                                                                  vocabulary_list=['test2','train2','eval2'],
                                                                  default_value=1 )
print(cate_eb)
eb_col=tf.feature_column.embedding_column(cate_eb,3)
eb_col1=tf.feature_column.embedding_column(cate_eb1,4)
eb_col2=tf.feature_column.embedding_column(cate_eb2,5)
print(eb_col)

inp_eb=tf.feature_column.input_layer(features_new,[eb_col,eb_col1,eb_col2])
# tf.reset_default_graph()
with tf.Session() as sess:
    # 在此处 必须使用 tf.tables_initializer来初始化 lookuptable
    sess.run([tf.global_variables_initializer(),tf.tables_initializer()])
#     sess.run()
    print(sess.run(inp_eb))
    print(inp_eb.shape)