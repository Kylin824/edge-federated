import json
import matplotlib.pyplot as plt
import numpy as np


def plot_raw_result():
    result_path = 'stats.txt'

    with open(result_path, 'r') as outfile:
        data = json.load(outfile)

    train_accuracy = data['train_accuracy']
    train_loss = data['train_loss']
    valid_accuracy = data['valid_accuracy']
    valid_loss = data['valid_loss']

    print(len(train_accuracy))
    print(len(valid_accuracy))

    x = []
    y = []
    t_acc = []
    v_acc = []
    num = []

    for r in train_accuracy:
        x.append(r[0])
        t_acc.append(r[2])

    for v in valid_accuracy:
        y.append(v[0])
        v_acc.append(v[2])
        num.append(v[3])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # 共享x轴
    ax1.plot(x, t_acc, color='r', label='train_acc')
    ax1.plot(y, v_acc, color='b', label='valid_acc')
    ax1.set(xlabel='round', ylabel='accuracy', title='ddqn4 niid choose')
    ax1.legend(loc=2)
    ax2.plot(y, num, color='g', label='client_num')
    ax2.set_ylim(0, 10)
    ax2.set(ylabel='update_num')
    ax2.legend(loc=7)
    plt.show()


def plot_raw_acc():
    nniid_50round_all_result = './result/mnist/nniid/50round_all/global_model_global_data_acc.txt'
    nniid_50round_drop3_result = './result/mnist/nniid/50round_drop3/global_model_global_data_acc.txt'

    iid_50round_all_result = './result/mnist/iid/50round_all/global_model_global_data_acc.txt'
    iid_50round_drop3_result = './result/mnist/iid/50round_drop3/global_model_global_data_acc.txt'

    niid_50round_all_result = './result/mnist/niid/50round_all/global_model_global_data_acc.txt'
    niid_50round_drop3_result = './result/mnist/niid/50round_drop3/global_model_global_data_acc.txt'


    nniid_50round_all_acc = np.loadtxt(nniid_50round_all_result)
    nniid_50round_drop3_acc = np.loadtxt(nniid_50round_drop3_result)

    iid_50round_all_acc = np.loadtxt(iid_50round_all_result)
    iid_50round_drop3_acc = np.loadtxt(iid_50round_drop3_result)

    niid_50round_all_acc = np.loadtxt(niid_50round_all_result)
    niid_50round_drop3_acc = np.loadtxt(niid_50round_drop3_result)

    x = np.arange(len(nniid_50round_all_acc))


    plt.plot(x, iid_50round_all_acc, label='iid_all_acc')
    plt.plot(x, iid_50round_drop3_acc, label='iid_drop3_acc')
    plt.plot(x, niid_50round_all_acc, label='niid_all_acc')
    plt.plot(x, niid_50round_drop3_acc, label='niid_drop3_acc')
    plt.plot(x, nniid_50round_all_acc, label='nniid_all_acc')
    plt.plot(x, nniid_50round_drop3_acc, label='nniid_drop3_acc')
    plt.ylim(0.9, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('round')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_raw_result()
    plot_raw_acc()
