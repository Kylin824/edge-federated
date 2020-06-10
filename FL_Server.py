import pickle
import keras
import uuid
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import random
import codecs
import numpy as np
import json
import time
from collections import defaultdict
from flask import *
from flask_socketio import *


class GlobalModel(object):
    """docstring for GlobalModel"""

    def __init__(self):
        self.model = self.build_model()
        self.current_weights = self.model.get_weights()
        # for convergence check
        self.prev_train_loss = None

        # all rounds; losses[i] = [round#, timestamp, loss]
        # round# could be None if not applicable
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.update_num = []

        self.training_start_time = int(round(time.time()))

    def build_model(self):
        raise NotImplementedError()

    # client_updates = [(w, n)..]
    def update_weights(self, client_weights, client_sizes):
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        total_size = np.sum(client_sizes)

        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
        self.current_weights = new_weights

    def aggregate_loss_accuracy(self, client_losses, client_accuracies, client_sizes):
        total_size = np.sum(client_sizes)
        # weighted sum
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                           for i in range(len(client_sizes)))
        aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
                                 for i in range(len(client_sizes)))
        return aggr_loss, aggr_accuraries

    # cur_round coule be None
    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round, update_num):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.train_losses += [[cur_round, cur_time, aggr_loss, update_num]]
        self.train_accuracies += [[cur_round, cur_time, aggr_accuraries, update_num]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    # cur_round coule be None
    def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round, update_num):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.valid_losses += [[cur_round, cur_time, aggr_loss, update_num]]
        self.valid_accuracies += [[cur_round, cur_time, aggr_accuraries, update_num]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss, aggr_accuraries

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies,
        }


class GlobalModel_MNIST_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()

    def build_model(self):
        # ~5MB worth of parameters
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])
        return model


######## Flask server with Socket IO ########
class FLServer(object):
    MIN_NUM_WORKERS = 3  # 超过多少个即可开始本轮训练
    MAX_NUM_ROUNDS = 10  # 最大轮次
    NUM_CLIENTS_CONTACTED_PER_ROUND = 3  # 每轮选多少个设备
    ROUNDS_BETWEEN_VALIDATIONS = 3

    def __init__(self, global_model, host, port):
        self.global_model = global_model()

        self.ready_client_sids = set()

        self.ordered_client = []

        self.client_index = -1
        self.ready_client_infos = defaultdict(list)

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port
        self.model_id = str(uuid.uuid4())

        self.mutex = 1
        self.observation = []

        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.current_round_valid_all = []
        self.eval_client_updates = []
        self.client_update_amount = 0
        #####

        # socket io messages
        self.register_handles()

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid, "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print(request.sid, "disconnected")
            # delete client from ready set
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)
                self.ordered_client.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
             # 01234
            self.client_index += 1
            client_index = (self.client_index + 0) % FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND
            emit('init', {
                'model_json': self.global_model.model.to_json(),
                'model_id': self.model_id,
                'min_train_size': 3000,  # 1200
                'data_split': (0.6, 0.3, 0.1),  # train, test, valid
                'epoch_per_round': 1,  # can be changed
                'batch_size': 50,
                'client_index': client_index
            })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            print("client ready for training", request.sid)
            print(data)
            self.client_update_amount += 1
            self.ready_client_sids.add(request.sid)
            self.ordered_client.append(request.sid)  # 按顺序
            self.ready_client_infos[request.sid].append(int(data['current_cq']))
            self.ready_client_infos[request.sid].append(int(data['pred_cq']))
            self.ready_client_infos[request.sid].append(int(data['current_cu']))
            self.ready_client_infos[request.sid].append(int(data['pred_cu']))
            print("ready client infos: ")
            print(self.ready_client_infos)
            # 达到开始训练最小数量要求
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1:
                print("client total amount: ", self.client_update_amount)

                first_observation = []
                # create first state
                for i in range(len(self.ordered_client)):
                    client_sid = self.ordered_client[i]
                    client_state = [self.ready_client_infos[client_sid][0],  # current_cq
                                    self.ready_client_infos[client_sid][1],  # pred_cq
                                    self.ready_client_infos[client_sid][2],  # current_cu
                                    self.ready_client_infos[client_sid][3]]  # pred_cu

                    first_observation.append(client_state)

                print("initial observation: ", first_observation)
                self.train_next_round(first_observation)

        @self.socketio.on('client_update')
        def handle_client_update(data):
            print("client update successfully: ", request.sid)

            # discard outdated update
            if data['round_number'] == self.current_round and request.sid in self.ready_client_sids:
                self.client_update_amount += 1
                print('already received: ', self.client_update_amount)

                # update cq & cq of client sid
                self.ready_client_infos[request.sid][0] = data['current_cq']
                self.ready_client_infos[request.sid][1] = data['pred_cq']
                self.ready_client_infos[request.sid][2] = data['current_cu']
                self.ready_client_infos[request.sid][3] = data['pred_cu']

                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])

                self.current_round_valid_all += [data]

                # 全部收到
                if self.client_update_amount >= FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND and \
                        len(self.current_round_client_updates) > 0 and self.mutex == 1:
                    self.mutex = 0

                    self.global_model.update_weights(
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                    )

                    aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        [x['train_accuracy'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round,
                        len(self.current_round_client_updates)
                    )

                    print('current update num: ', len(self.current_round_client_updates))
                    print("aggr_train_loss", aggr_train_loss)
                    print("aggr_train_accuracy", aggr_train_accuracy)

                    if self.current_round_valid_all:
                        if 'valid_loss' in self.current_round_valid_all[0]:
                            aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                                [x['valid_loss'] for x in self.current_round_valid_all],
                                [x['valid_accuracy'] for x in self.current_round_valid_all],
                                [x['valid_size'] for x in self.current_round_valid_all],
                                self.current_round,
                                len(self.current_round_client_updates)
                            )
                            print("aggr_valid_loss", aggr_valid_loss)
                            print("aggr_valid_accuracy", aggr_valid_accuracy)

                    self.global_model.prev_train_loss = aggr_train_loss

                    # final round
                    if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                        self.stop_and_eval()
                    # need next round  update_type=0
                    else:
                        print('1111111111       ', self.client_update_amount)
                        observation_ = self.after_action()
                        print('observation: ', observation_)
                        self.train_next_round(observation_)

        @self.socketio.on('not_client_update')
        def handle_not_client_update(data):
            # print("received not client update of bytes: ", sys.getsizeof(data))
            print("handle not client_update", request.sid)

            if data['round_number'] == self.current_round and request.sid in self.ready_client_sids:

                self.client_update_amount += 1
                print('already received: ', self.client_update_amount)

                self.current_round_valid_all += [data]

                # update cq & cq of client sid
                self.ready_client_infos[request.sid][0] = data['current_cq']
                self.ready_client_infos[request.sid][1] = data['pred_cq']
                self.ready_client_infos[request.sid][2] = data['current_cu']
                self.ready_client_infos[request.sid][3] = data['pred_cu']

                # print("not client infos: ")
                # print(self.ready_client_infos)

                # 全部收到
                if self.client_update_amount >= FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:

                    if self.current_round_valid_all:
                        if 'valid_loss' in self.current_round_valid_all[0]:
                            aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                                [x['valid_loss'] for x in self.current_round_valid_all],
                                [x['valid_accuracy'] for x in self.current_round_valid_all],
                                [x['valid_size'] for x in self.current_round_valid_all],
                                self.current_round,
                                len(self.current_round_client_updates)
                            )
                            print("aggr_valid_loss", aggr_valid_loss)
                            print("aggr_valid_accuracy", aggr_valid_accuracy)

                    if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                        self.stop_and_eval()
                    else:
                        print('22222222222    : ', self.client_update_amount)
                        observation_ = self.after_action()
                        self.train_next_round(observation_)

        @self.socketio.on('client_timeout')
        def handle_client_timeout(data):
            # print("received timeout update of bytes: ", sys.getsizeof(data))
            print("client timeout sid:", request.sid)

            if data['round_number'] == self.current_round:

                self.client_update_amount += 1
                print('already received: ', self.client_update_amount)

                self.current_round_valid_all += [data]

                # update cq & cq of client sid
                self.ready_client_infos[request.sid][0] = data['current_cq']
                self.ready_client_infos[request.sid][1] = data['pred_cq']
                self.ready_client_infos[request.sid][2] = data['current_cu']
                self.ready_client_infos[request.sid][3] = data['pred_cu']

                # 全部收到
                if self.client_update_amount >= FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:

                    if self.current_round_valid_all:

                        if 'valid_loss' in self.current_round_valid_all[0]:
                            aggr_valid_loss, aggr_valid_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                                [x['valid_loss'] for x in self.current_round_valid_all],
                                [x['valid_accuracy'] for x in self.current_round_valid_all],
                                [x['valid_size'] for x in self.current_round_valid_all],
                                self.current_round,
                                len(self.current_round_client_updates)
                            )
                            print("aggr_valid_loss", aggr_valid_loss)
                            print("aggr_valid_accuracy", aggr_valid_accuracy)

                    if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                        self.stop_and_eval()

                    # already receive all clients, turn to next round
                    else:
                        print('33333333333      ', self.client_update_amount)
                        observation_ = self.after_action()
                        self.train_next_round(observation_)

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            print("handle client_eval", request.sid)
            print("eval_resp", data)
            self.eval_client_updates += [data]

            # tolerate 30% unresponsive clients
            if len(self.eval_client_updates) >= len(self.ordered_client):
                aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_loss_accuracy(
                    [x['test_loss'] for x in self.eval_client_updates],
                    [x['test_accuracy'] for x in self.eval_client_updates],
                    [x['test_size'] for x in self.eval_client_updates],
                )
                print("\naggr_test_loss", aggr_test_loss)
                print("aggr_test_accuracy", aggr_test_accuracy)
                print("== done ==")
                self.eval_client_updates = None  # special value, forbid evaling again

    def select_rand_client(self, ready_client_sids):
        client_sids_selected = random.sample(list(ready_client_sids), FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND)
        return client_sids_selected

    def select_greedy_client(self, ready_client_sids):
        client_sids_selected = list(ready_client_sids)
        return client_sids_selected

    def select_FedCS_client(self, ready_client_sids):
        client_sids_selected = []
        for sid in list(ready_client_sids):
            if len(client_sids_selected) < FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:
                if int(self.ready_client_infos[sid][0]) >= 4 and \
                        int(self.ready_client_infos[sid][2]) >= 4:
                    client_sids_selected.append(sid)

        # # 符合条件的数量不够
        # if len(client_sids_selected) < FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:
        #     for sid in list(ready_client_sids):
        #         if sid not in client_sids_selected and \
        #                 len(client_sids_selected) < FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:
        #             client_sids_selected.append(sid)

        return client_sids_selected

    # def select_drl_client(self, observation):
    #
    #     actions_value = ddqn_model.predict(observation)
    #     action = np.argmax(actions_value)
    #     client_sids_selected = []
    #     bin = '{:010b}'.format(action)
    #     if bin[0] == '1':
    #         client_sids_selected.append(self.ordered_client[0])
    #     if bin[1] == '1':
    #         client_sids_selected.append(self.ordered_client[1])
    #     if bin[2] == '1':
    #         client_sids_selected.append(self.ordered_client[2])
    #     if bin[3] == '1':
    #         client_sids_selected.append(self.ordered_client[3])
    #     if bin[4] == '1':
    #         client_sids_selected.append(self.ordered_client[4])
    #     if bin[5] == '1':
    #         client_sids_selected.append(self.ordered_client[5])
    #     if bin[6] == '1':
    #         client_sids_selected.append(self.ordered_client[6])
    #     if bin[7] == '1':
    #         client_sids_selected.append(self.ordered_client[7])
    #     if bin[8] == '1':
    #         client_sids_selected.append(self.ordered_client[8])
    #     if bin[9] == '1':
    #         client_sids_selected.append(self.ordered_client[9])
    #     return client_sids_selected

    def select_fedOpt_client(self, ready_client_sids):

        client_sids_selected = []
        for sid in list(ready_client_sids):
            if len(client_sids_selected) < FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:

                total_cq = int(self.ready_client_infos[sid][0]) + int(self.ready_client_infos[sid][1])
                total_cu = int(self.ready_client_infos[sid][2]) + int(self.ready_client_infos[sid][3])

                if total_cq >= 6 and total_cu >= 6:
                    client_sids_selected.append(sid)

        # # 符合条件的数量不够
        # if len(client_sids_selected) < FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:
        #     for sid in list(ready_client_sids):
        #         if sid not in client_sids_selected and \
        #                 len(client_sids_selected) < FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND:
        #             client_sids_selected.append(sid)

        return client_sids_selected

    def after_action(self):
        observation_ = []

        for i in range(len(self.ordered_client)):
            client_sid = self.ordered_client[i]
            # print('client:', client_sid)
            client_state = [self.ready_client_infos[client_sid][0],  # current_cq
                            self.ready_client_infos[client_sid][1],  # pred_cq
                            self.ready_client_infos[client_sid][2],  # current_cu
                            self.ready_client_infos[client_sid][3]]  # pred_cu
            observation_.append(client_state)
            # print('state: ', client_state)
        print("observation_ : ", observation_)
        return observation_

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self, observation):
        self.current_round += 1
        # buffers all client updates

        self.current_round_client_updates = []
        self.current_round_valid_all = []
        self.client_update_amount = 0

        self.mutex = 1

        print("### Round ", self.current_round, "###")

        # randomly choose ready client
        client_sids_selected = self.select_rand_client(self.ready_client_sids)

        # fedCS choose client
        # client_sids_selected = self.select_FedCS_client(self.ready_client_sids)

        # opt choose client
        # client_sids_selected = self.select_fedOpt_client(self.ready_client_sids)

        # ddqn choose client
        # observation = np.array(observation).reshape(1, -1)  # from [[..],[..],[..],[..]] to [[.................]]
        # client_sids_selected = self.select_drl_client(observation)

        print("request updates from", client_sids_selected)

        for rid in self.ready_client_sids:
            if rid in client_sids_selected:
                emit('request_update', {
                    'is_selected': 'true',
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle',
                    'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }, room=rid)
            else:
                emit('request_update', {
                    'is_selected': 'false',
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle',
                    'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }, room=rid)

    def stop_and_eval(self):
        self.eval_client_updates = []
        for rid in self.ready_client_sids:
            emit('stop_and_eval', {
                'model_id': self.model_id,
                'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                'weights_format': 'pickle'
            }, room=rid)

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)


def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()


def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))


if __name__ == '__main__':

    server = FLServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000)
    print("listening on 127.0.0.1:5000")
    server.start()
