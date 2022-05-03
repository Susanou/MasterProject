url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
wget ${url}
tar -xvzf cifar-100-python.tar.gz
rm cifar-100-python.tar.gz
python2 preprocess_cifar100.py
python2 create_cifar100_train.py
rm -rf cifar-100-python/
