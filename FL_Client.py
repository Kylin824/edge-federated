import numpy as np
import keras
import random
import time
from keras.models import model_from_json
from socketIO_client import SocketIO, LoggingNamespace
from FL_Server import obj_to_pickle_string, pickle_string_to_obj
import datasource
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class LocalModel(object):
    def __init__(self, model_config, data_collected):
        self.model_config = model_config

        self.model = model_from_json(model_config['model_json'])

        # MNIST
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.SGD(),
                           #  optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

        train_data, test_data, valid_data = data_collected

        self.x_train = np.array([tup[0] for tup in train_data])
        self.y_train = np.array([tup[1] for tup in train_data])
        self.x_test = np.array([tup[0] for tup in test_data])
        self.y_test = np.array([tup[1] for tup in test_data])
        self.x_valid = np.array([tup[0] for tup in valid_data])
        self.y_valid = np.array([tup[1] for tup in valid_data])

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def train_one_round(self):

        # mnist
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.SGD(),
                           metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train,
                       epochs=self.model_config['epoch_per_round'],
                       batch_size=self.model_config['batch_size'],
                       verbose=0,
                       shuffle=True,
                       validation_data=(self.x_valid, self.y_valid))

        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return self.model.get_weights(), score[0], score[1]

    def validate(self):
        score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        # print('Validate loss:', score[0])
        print('Validate accuracy:', score[1])
        return score

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score


class FederatedClient(object):

    # 每个client的最大数据量
    # MAX_DATASET_SIZE_KEPT = 3000

    def __init__(self, server_host, server_port, datasource, index):
        self.local_model = None
        self.datasource = datasource()
        self.sio = SocketIO(server_host, server_port, LoggingNamespace)
        self.index = index
        self.state_transition = pd.read_csv('./client_state/10client_ob_state.csv').values
        self.real_state = pd.read_csv('./client_state/10client_real_state.csv').values
        self.register_handles()
        self.prev_train_loss = 0
        self.prev_train_acc = 0
        self.global_model_local_data_acc = []
        self.global_model_global_data_acc = []
        self.local_model_local_data_acc = []
        self.local_model_global_data_acc = []
        self.drop_every_round = 3
        print("sent wakeup")  # 通知server
        self.sio.emit('client_wake_up')
        self.sio.wait()

    ########## Socket Event Handler ##########

    def get_pred_state(self, client_index, round_num):
        cur_cq = str(self.state_transition[round_num][4*client_index])
        pred_cq = str(self.state_transition[round_num][4*client_index+1])
        cur_cu = str(self.state_transition[round_num][4*client_index+2])
        pred_cu = str(self.state_transition[round_num][4*client_index+3])
        return cur_cq, pred_cq, cur_cu, pred_cu

    def get_real_state(self, client_index, round_num):
        next_cq = str(self.real_state[round_num][2*client_index])
        next_cu = str(self.real_state[round_num][2*client_index+1])
        return next_cq, next_cu

    def on_init(self, *args):
        model_config = args[0]
        print('on init', model_config)  # model_config为server传来的参数
        print('preparing local data based on server model_config')

        fake_data, my_class_distr = self.datasource.load_local_iid_data(model_config['client_index'])

        print('done load local client_dataset')
        print(my_class_distr)

        # load 本地的数据集
        self.local_model = LocalModel(model_config, fake_data)

        cur_cq, pred_cq, cur_cu, pred_cu = self.get_pred_state(round_num=0, client_index=self.index)

        init_state = [cur_cq, pred_cq, cur_cu, pred_cu]

        print('init state: ', init_state)

        # ready to be dispatched for training
        self.sio.emit('client_ready', {
                'train_size': self.local_model.x_train.shape[0],
                'class_distr': my_class_distr,  # for debugging, not needed in practice
                'current_cq': cur_cq,
                'pred_cq': pred_cq,
                'current_cu': cur_cu,
                'pred_cu': pred_cu
            })

    def register_handles(self):
        ########## Socket IO messaging ##########
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        # client上传一轮的训练结果
        def on_request_update(*args):
            req = args[0]

            # self.local_model = LocalModel(model_config, fake_data)
            round_num = req['round_number'] % 199

            cur_cq, pred_cq, cur_cu, pred_cu = self.get_pred_state(round_num=round_num, client_index=self.index)
            next_cq, next_cu = self.get_real_state(round_num=round_num, client_index=self.index)

            total_cq = int(cur_cq) + int(next_cq)
            total_cu = int(cur_cu) + int(next_cu)

            cur_cq, pred_cq, cur_cu, pred_cu = self.get_pred_state(round_num=round_num+1, client_index=self.index)

            if req['is_selected'] == 'true':
                if req['weights_format'] == 'pickle':
                    weights = pickle_string_to_obj(req['current_weights'])
                    self.local_model.set_weights(weights)

                # 测试下载的global model在client上的精度
                test_loss, local_acc = self.local_model.evaluate()
                valid_loss, global_acc = self.local_model.validate()
                print('round: ' + str(round_num) + ' -> global model on local dataset acc: ' + str(local_acc))
                print('round: ' + str(round_num) + ' -> global model on valid dataset acc: ' + str(global_acc))

                self.global_model_local_data_acc.append(local_acc)
                self.global_model_global_data_acc.append(global_acc)

                if (self.index == 2 and req['round_number'] % self.drop_every_round == 0):
                    resp = {
                        'round_number': req['round_number'],
                        'weights': req['current_weights'],
                        'train_size': self.local_model.x_train.shape[0],
                        'valid_size': self.local_model.x_valid.shape[0],
                        'train_loss': self.prev_train_loss,
                        'train_accuracy': self.prev_train_acc,
                        'current_cq': cur_cq,
                        'pred_cq': pred_cq,
                        'current_cu': cur_cu,
                        'pred_cu': pred_cu
                    }
                    print("\nclient drop at round: ", req['round_number'])

                else:
                    # 训练一轮
                    my_weights, train_loss, train_accuracy = self.local_model.train_one_round()
                    self.prev_train_loss = train_loss
                    self.prev_train_acc = train_accuracy

                    resp = {
                        'round_number': req['round_number'],
                        'weights': obj_to_pickle_string(my_weights),
                        'train_size': self.local_model.x_train.shape[0],
                        'valid_size': self.local_model.x_valid.shape[0],
                        'train_loss': train_loss,
                        'train_accuracy': train_accuracy,
                        'current_cq': cur_cq,
                        'pred_cq': pred_cq,
                        'current_cu': cur_cu,
                        'pred_cu': pred_cu
                    }
                    print("\nsuccessfully update at round: ", req['round_number'])

                if req['run_validation']:
                    valid_loss, valid_accuracy = self.local_model.validate()
                    resp['valid_loss'] = valid_loss
                    resp['valid_accuracy'] = valid_accuracy

                self.sio.emit('client_update', resp)

            #
            # if req['is_selected'] == 'true':
            #
            #     # good to updated
            #     if total_cq >= 6 and total_cu >= 6:
            #
            #
            #         if req['weights_format'] == 'pickle':
            #             weights = pickle_string_to_obj(req['current_weights'])
            #         self.local_model.set_weights(weights)
            #
            #         # 训练一轮
            #         my_weights, train_loss, train_accuracy = self.local_model.train_one_round()
            #         self.prev_train_loss = train_loss
            #         self.prev_train_acc = train_accuracy
            #
            #         resp = {
            #             'round_number': req['round_number'],
            #             'weights': obj_to_pickle_string(my_weights),
            #             'train_size': self.local_model.x_train.shape[0],
            #             'valid_size': self.local_model.x_valid.shape[0],
            #             'train_loss': train_loss,
            #             'train_accuracy': train_accuracy,
            #             'current_cq': cur_cq,
            #             'pred_cq': pred_cq,
            #             'current_cu': cur_cu,
            #             'pred_cu': pred_cu
            #         }
            #         print("\nsuccessfully update at round: ", req['round_number'])
            #         if req['run_validation']:
            #             valid_loss, valid_accuracy = self.local_model.validate()
            #             resp['valid_loss'] = valid_loss
            #             resp['valid_accuracy'] = valid_accuracy
            #
            #         self.sio.emit('client_update', resp)
            #
            #     # selectd but timeout
            #     else:
            #         if req['weights_format'] == 'pickle':
            #             weights = pickle_string_to_obj(req['current_weights'])
            #         self.local_model.set_weights(weights)
            #
            #         resp = {
            #             'round_number': req['round_number'],
            #             'train_size': self.local_model.x_train.shape[0],
            #             'valid_size': self.local_model.x_valid.shape[0],
            #             'train_loss': self.prev_train_loss,
            #             'train_accuracy': self.prev_train_acc,
            #             'current_cq': cur_cq,
            #             'pred_cq': pred_cq,
            #             'current_cu': cur_cu,
            #             'pred_cu': pred_cu
            #         }
            #         print("\nclient timeout at round: ", req['round_number'])
            #         if req['run_validation']:
            #             valid_loss, valid_accuracy = self.local_model.validate()
            #             resp['valid_loss'] = valid_loss
            #             resp['valid_accuracy'] = valid_accuracy
            #
            #
            #         self.sio.emit('client_timeout', resp)
            # # not selected
            # else:
            #     if req['weights_format'] == 'pickle':
            #         weights = pickle_string_to_obj(req['current_weights'])
            #     self.local_model.set_weights(weights)
            #
            #     resp = {
            #         'round_number': req['round_number'],
            #         'train_size': self.local_model.x_train.shape[0],
            #         'valid_size': self.local_model.x_valid.shape[0],
            #         'train_loss': self.prev_train_loss,
            #         'train_accuracy': self.prev_train_acc,
            #         'current_cq': cur_cq,
            #         'pred_cq': pred_cq,
            #         'current_cu': cur_cu,
            #         'pred_cu': pred_cu
            #     }
            #     print("\nnot selected at round: ", req['round_number'])
            #     if req['run_validation']:
            #         valid_loss, valid_accuracy = self.local_model.validate()
            #         resp['valid_loss'] = valid_loss
            #         resp['valid_accuracy'] = valid_accuracy
            #
            #
            #     self.sio.emit('not_client_update', resp)

        # client上传测试精度
        def on_stop_and_eval(*args):
            req = args[0]

            valid_loss, valid_accuracy = self.local_model.validate()
            print('final local model on global dataset acc: ' + str(valid_accuracy))


            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])

            self.local_model.set_weights(weights)
            test_loss, test_accuracy = self.local_model.evaluate()
            print('final global model on local dataset acc: ' + str(test_accuracy))

            print('init result.txt')

            acc_file_dir = 'result/mnist/nniid/'

            local_acc_file = acc_file_dir + 'client_' + str(self.index) + '_global_model_local_data_acc.txt'
            with open(local_acc_file, 'a') as lf:
                for i in self.global_model_local_data_acc:
                    lf.write(str(i)+'\n')

            if (self.index == 0):
                global_acc_file = acc_file_dir + 'global_model_global_data_acc.txt'
                with open(global_acc_file, 'a') as gf:
                    for i in self.global_model_global_data_acc:
                        gf.write(str(i) + '\n')

            resp = {
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }

            self.sio.emit('client_eval', resp)

        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', lambda *args: self.on_init(*args))
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)

    # clients间歇地休眠
    def intermittently_sleep(self, p=.1, low=10, high=100):
        if (random.random() < p):
            time.sleep(random.randint(low, high))


if __name__ == "__main__":
    FederatedClient("127.0.0.1", 5000, datasource.Mnist, 0)
    # FederatedClient("127.0.0.1", 5000, datasource.CIFAR, 2)






