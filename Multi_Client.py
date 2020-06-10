from FL_Client import FederatedClient
import datasource
import multiprocessing
import time

def start_client(client_index):
    print("start client")
    FederatedClient("127.0.0.1", 5000, datasource.Mnist, client_index)
    # FederatedClient("192.168.199.179", 5000, datasource.Mnist, client_index)
    # FederatedClient("127.0.0.1", 5000, datasource.CIFAR, client_index)
    # FederatedClient("127.0.0.1", 5000, datasource.FashionMnist)


if __name__ == '__main__':
    # jobs = []
    multiprocessing.freeze_support()
    for i in range(3):
        p = multiprocessing.Process(target=start_client, args=(i,))
        # jobs.append(p)
        p.start()
        time.sleep(5)
    # p.join()
    # print('end')
