# -*- coding: utf-8 -*-
# @Time    : 2020/9/5 23:39
# @Author  : Zeqi@@
# @FileName: download_cifar10_win.py
# @Software: PyCharm


import urllib.request
import os
import tarfile

# Download
os.makedirs('../data_download', exist_ok=True)
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
filepath = '../data_download/cifar-10-python.tar.gz'
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url, filepath)
    print('download:', result)
else:
    print('Data file already exists.')

# Extracting
extract_directory = '../data_download/cifar-10-python'
if not os.path.exists('../data_download/cifar-10-python'):
    tfile = tarfile.open(filepath, 'r:gz')
    result = tfile.extractall('../data_download/cifar-10-python')
    print("Extracting successfully done to {}.".format(extract_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")