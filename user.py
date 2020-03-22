#import tensorflow as tf
import collections

class User():
    def __init__(self,name,age):
        self._name = name
        self._age = age

    @property
    def get_name(self):
        return self._name

def b():
    return 3



if __name__ == '__main__':
    user = User('a',1)
    print(user.get_name)
    print(b())
    #tf.compat.v1.disable_eager_execution()
    #a = tf.constant(1.0)
    #b = tf.constant(2.0)
    # with tf.compat.v1.Session().as_default() as sess:
    #     print(a.eval())
    #
    # print(b.eval(session=sess))
    metric = ({})
    print(len(metric))
    metric = ({},)
    print(len(metric))
    metric = ({},...)
    print(len(metric))

    a = []
    print(len(a))
    a = [()]
    print(len(a))
    a = [(),]
    print(len(a))

    metric_vals = [collections.defaultdict(list) for _ in metric]
    results = [{'train_error': 15.572575, 'test_error': 15.675269}]
    print(metric_vals)
    print(list(zip(metric_vals,results)))
    for metric_val,result in zip(metric_vals,results):
        for k ,v in result.items():
            metric_val[k].append(v)
    print(metric_vals)