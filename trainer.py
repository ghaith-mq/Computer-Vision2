#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[40]:


import joblib
import random
from sklearn.neural_network import MLPClassifier
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# pip install python-mnist
from mnist import MNIST
import warnings
# download 4 MNIST files you can here:
# http://yann.lecun.com/exdb/mnist/


MNIST_CELL_SIZE = 28
SEED = 0xBadCafe


def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(save_model_path='/autograder/submission/semi1.joblib', mnist_data_path='/autograder/submission/', print_accuracy=True):

    # FIXING RANDOM SEED:
    fix_seed(SEED)

    # EXAMPLE OF LOADING MNIST
    mnist_data = MNIST(mnist_data_path)
    mnist_data.gz = True

    images_train, labels_train = mnist_data.load_training()
    images_test, labels_test = mnist_data.load_testing()

    images_train = np.uint8([np.reshape(im, (MNIST_CELL_SIZE,) * 2) for im in images_train])
    images_test = np.uint8([np.reshape(im, (MNIST_CELL_SIZE,) * 2) for im in images_test])
    labels_train, labels_test = np.int16(labels_train), np.int16(labels_test)

    # EXAMPLE OF TRAINING RandomForest

    # hog = get_hog()  # from cv2
    # features_train = np.array([hog.compute(im).T[0] for im in images_train])
    # features_test = np.array([hog.compute(im).T[0] for im in images_test])
    features_train = np.array([im.ravel() for im in images_train])
    features_test = np.array([im.ravel() for im in images_test])
    clf = MLPClassifier(random_state=1, max_iter=300)

# this example won't converge because of CI's time constraints, so we catch the
# warning and are ignore it here
    clf.fit(features_train, labels_train)

    # SAVING MODEL !!!
    joblib.dump(clf, save_model_path)
    #joblib.load('nn.joblib')

    if print_accuracy:
        from sklearn.metrics import accuracy_score
        print(accuracy_score(labels_test, clf.predict(features_test)))
    


# In[41]:


train()


# In[ ]:




