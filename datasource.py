import numpy as np
import keras
import random
from keras.datasets import mnist
from keras import backend as K
import os

class DataSource(object):
    def __init__(self):
        raise NotImplementedError()
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        raise NotImplementedError()
    def sample_single_non_iid(self, weight=None):
        raise NotImplementedError()
    def load_local_iid_data(self, client_index):
        raise NotImplementedError()


class Mnist(DataSource):

    IID = True  # True: generate iid dataset  False: generate non-iid dataset
    MIN_NUM_CLASSES_PER_CLIENT = 3
    MAX_NUM_CLASSES_PER_CLIENT = 3
    
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x = np.concatenate([x_train, x_test]).astype('float')
        self.y = np.concatenate([y_train, y_test])
        n = self.x.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        self.x = self.x[idx]  # n * 28 * 28
        self.y = self.y[idx]  # n * 1
        data_split = (0.6, 0.3, 0.1)
        num_train = int(n * data_split[0])
        num_test = int(n * data_split[1])
        self.x_train = self.x[0:num_train]
        self.x_test = self.x[num_train:num_train + num_test]
        self.x_valid = self.x[num_train + num_test:]
        self.y_train = self.y[0:num_train]
        self.y_test = self.y[num_train:num_train + num_test]
        self.y_valid = self.y[num_train + num_test:]
        self.classes = np.unique(self.y)
    
    def gen_dummy_non_iid_weights(self):
        self.classes = np.array(range(10))

        num_classes_this_client = 3
        classes_this_client = [4, 5, 7]

        w = np.array([random.random() for _ in range(num_classes_this_client)])
        weights = np.array([0.] * self.classes.shape[0])
        for i in range(len(classes_this_client)):
            weights[classes_this_client[i]] = w[i]
        weights /= np.sum(weights)
        return weights.tolist()

    # assuming client server already agreed on data format
    def post_process(self, xi, yi):
        if K.image_data_format() == 'channels_first':
            xi = xi.reshape(1, xi.shape[0], xi.shape[1])
        else:
            xi = xi.reshape(xi.shape[0], xi.shape[1], 1)

        y_vec = keras.utils.to_categorical(yi, self.classes.shape[0])
        return xi / 255., y_vec

    # split evenly into exact num_workers chunks, with test_reserve globally
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        n_test = int(self.x.shape[0] * test_reserve)
        n_train = self.x.shape[0] - n_test
        nums = [n_train // num_workers] * num_workers
        nums[-1] += n_train % num_workers
        idxs = np.array([np.random.choice(np.arange(n_train), num, replace=False) for num in nums])
        return {
            # (size_partition * 28 * 28, size_partition * 1) * num_partitions
            "train": [self.post_process(self.x[idx], self.y[idx]) for idx in idxs],
            # (n_test * 28 * 28, n_test * 1)
            "test": self.post_process(self.x[np.arange(n_train, n_train + n_test)], self.y[np.arange(n_train, n_train + n_test)])
        }

    # Generate one sample from all available data, *with replacement*.
    # This is to simulate date generation on a client.
    # weight: [probablity of classes]
    # returns: 28 * 28, 1
    def sample_single_non_iid(self, x, y, weight=None):
        # first pick class, then pick a datapoint at random
        chosen_class = np.random.choice(self.classes, p=weight)
        candidates_idx = np.array([i for i in range(y.shape[0]) if y[i] == chosen_class])
        idx = np.random.choice(candidates_idx)
        return self.post_process(x[idx], y[idx])

    # generate t, t, v client_dataset given distribution and split
    def fake_non_iid_data(self, min_train=100, max_train=1000, data_split=(.6, .3, .1)):
        # my_class_distr = np.array([np.random.random() for _ in range(self.classes.shape[0])])
        # my_class_distr /= np.sum(my_class_distr)
        my_class_distr = [1. / self.classes.shape[0]] * self.classes.shape[0] if Mnist.IID \
                else self.gen_dummy_non_iid_weights()

        train_size = random.randint(min_train, max_train)
        print('train size: ', train_size)
        test_size = int(train_size / data_split[0] * data_split[1])
        valid_size = int(train_size / data_split[0] * data_split[2])

        train_set = [self.sample_single_non_iid(self.x_train, self.y_train, my_class_distr) for _ in range(train_size)]
        test_set = [self.sample_single_non_iid(self.x_test, self.y_test, my_class_distr) for _ in range(test_size)]
        valid_set = [self.sample_single_non_iid(self.x_valid, self.y_valid, my_class_distr) for _ in range(valid_size)]
        print("done generating fake data")

        return (train_set, test_set, valid_set), my_class_distr

    def load_local_iid_data(self, client_index):
        # client_index = 01234
        data_dir = 'client_dataset/mnist/iid/'
        size = '10000';
        train_data_name = 'train_c' + str(client_index) + '_' + size + '.npy'
        train_data_path = os.path.join(data_dir, train_data_name)
        test_data_name = 'test_c' + str(client_index) + '_' + size + '.npy'
        test_data_path = os.path.join(data_dir, test_data_name)
        valid_data_name = 'valid_c' + str(client_index) + '_' + size + '.npy'
        valid_data_path = os.path.join(data_dir, valid_data_name)
        distri_name = 'distribution_c' + str(client_index) + '_' + size + '.txt'
        distri_path = os.path.join(data_dir, distri_name)

        train_set = np.load(train_data_path, allow_pickle=True)
        test_set = np.load(test_data_path, allow_pickle=True)
        valid_set = np.load(valid_data_path, allow_pickle=True)
        class_distr = np.loadtxt(distri_path).tolist()

        return (train_set, test_set, valid_set), class_distr


if __name__ == "__main__":
    m = Mnist()
    (train_set, test_set, valid_set), class_distr = m.fake_non_iid_data(min_train=10000, max_train=10000)

    train = np.array(train_set)
    test = np.array(test_set)
    valid = np.array(valid_set)

    data_dir = 'client_dataset/mnist/iid/'
    size = '10000'
    client_index = 4

    np.save('train_c' + str(client_index) + '_' + size + '.npy', train)
    # 'client_dataset/mnist/iid/train_c1_10000.npy'
    np.save('test_c' + str(client_index) + '_' + size + '.npy', test)
    np.save('valid_c' + str(client_index) + '_' + size + '.npy', valid)
    np.savetxt('distribution_c' + str(client_index) + '_' + size + '.txt', class_distr)
    print(class_distr)

