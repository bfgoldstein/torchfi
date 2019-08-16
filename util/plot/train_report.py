import numpy as np
import matplotlib.pyplot as plt


def plot_losses(fname, train_losses, fi_train_losses, test_losses, fi_test_losses):
    plt.figure(1, figsize = (14, 12))

    plt.subplot(211)
    plt.title('train')
    plt.plot(np.arange(len(train_losses)), np.array(train_losses), 'b', label='Golden loss')
    # plt.plot(np.arange(len(train_accs)), np.array(train_accs), 'g--', label='accuracy')
    plt.plot(np.arange(len(fi_train_losses)), np.array(fi_train_losses), 'r', label='Faulty loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid()

    plt.subplot(212)
    plt.title('test')
    plt.plot(np.arange(len(test_losses)), np.array(test_losses), 'r', label='Golden loss')
    # plt.plot(np.arange(len(test_accs)), np.array(test_accs), 'g--', label='accuracy')
    plt.plot(np.arange(len(fi_test_losses)), np.array(fi_test_losses), 'm', label='Faulty loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig(fname + '_losses.eps', dpi=300, bbox_inches='tight', format='eps')