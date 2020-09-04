# import copy
import os
from subprocess import call

print("")

# Downloading data
print("Downloading...")

if not os.path.exists("../data_download/cifar-10-python.tar.gz"):
    call(
        'mkdir "../data_download"',
        shell=True
    )
    call(
        'wget -O "../data_download/cifar-10-python.tar.gz" "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")

# Extracting tar data
print("Extracting...")
extract_directory = os.path.abspath("../data_download/cifar-10-python")
print(extract_directory)
if not os.path.exists(extract_directory):
    call(
        'mkdir "../data_download/cifar-10-python"',
        shell=True
    )
    call(
    'tar zxvf "../data_download/cifar-10-python.tar.gz" -C "../data_download/cifar-10-python"',
    shell=True
    )
    print("Extracting successfully done to {}.".format(extract_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")
