# -*- coding: utf-8 -*-
# @Time    : 2020/9/5 22:34
# @Author  : Zeqi@@
# @FileName: utils.py
# @Software: PyCharm


import matplotlib.pyplot as plt

def acc_monitoring(epoch,
                   train_loss,
                   test_loss,
                   train_accuracy,
                   test_accuracy,
                   save_path):
    ### Train: Green(#15b01a), aqua(#13eac9), sea green(#53fca1), aquamarine, grass green(#3f9b0b), forest green(#154406), mint green(#9ffeb0), spring green(#a9f971)
    ### Test: Blue, light blue(#95d0fc), sky blue(#448ee4), royal blue(#0504aa), bright blue(#0165fc),  pale blue(#d0fefe), electric blue(#0652ff), darker blue(#00035b)
    ### Sat_loss: Yellow, bright yellow(#fffd01), golden yellow(#fac205), lemon(#fdff52), shit
    ### Deep layer:
    ### Sallow layer:


    plt.figure()
    plt.subplot(211)
    plt.plot(epoch, train_accuracy, label="$Train \quad Accuracy$", color="blue", linewidth=0.5)
    plt.plot(epoch, test_accuracy, label="$Test \quad Accuracy$", color="green", linewidth=0.5, linestyle='--')
    plt.legend(loc='upper left', fontsize=8)
    plt.ylabel('Accuracy')

    plt.subplot(212)
    plt.plot(epoch, train_loss, label="$Train \quad loss$", color="#448ee4", linewidth=0.5, linestyle= '-')
    plt.plot(epoch, test_loss, label="$Test \quad loss$", color="#53fca1", linewidth=0.5, linestyle='--')
    plt.legend(loc='upper left', fontsize=8)
    # plt.ylim((0, 0.002))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    fig = plt.gcf()
    fig.savefig(save_path, dpi=600)
    plt.close()
