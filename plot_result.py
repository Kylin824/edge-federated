import json
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    plot_raw_result()


