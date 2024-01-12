{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "korean-fancy",
   "metadata": {},
   "source": [
    "# Image Classification with HOG Feature and Support Vector Machine "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "welcome-thesaurus",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from skimage.feature import hog\n",
    "from sklearn import datasets\n",
    "from mlxtend.data import loadlocal_mnist\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "from sklearn.svm import SVC"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "laden-franklin",
   "metadata": {},
   "source": [
    "## 1. Dataset Preparation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "involved-botswana",
   "metadata": {},
   "source": [
    "**Load image dataset**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d53a2184",
   "metadata": {},
   "outputs": [],
   "source": [
    "images_path = '/Users/teknikelektro/Documents/MACHINE VISION /train-images.idx3-ubyte'\n",
    "labels_path = '/Users/teknikelektro/Documents/MACHINE VISION /train-labels.idx1-ubyte'\n",
    "train_images, train_labels = loadlocal_mnist(images_path=images_path, labels_path=labels_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fixed-norwegian",
   "metadata": {},
   "outputs": [],
   "source": [
    "images_path = '/Users/teknikelektro/Documents/MACHINE VISION /t10k-images.idx3-ubyte'\n",
    "labels_path = '//Users/teknikelektro/Documents/MACHINE VISION /t10k-labels.idx1-ubyte'\n",
    "test_images, test_labels = loadlocal_mnist(images_path=images_path, labels_path=labels_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "chief-exhibition",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x29ea1afd0>"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(train_images[90].reshape(28,28), cmap='gray')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "international-router",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "6"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_labels[90]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "statistical-medicaid",
   "metadata": {},
   "source": [
    "## 2. Preprocessing using HOG Feature Extraction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "b5ae1a91",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature, hog_img = hog(train_images[0].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "magnetic-external",
   "metadata": {},
   "outputs": [],
   "source": [
    "n_dims = feature.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "twenty-allah",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "144"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n_dims"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "statutory-photography",
   "metadata": {},
   "outputs": [],
   "source": [
    "n_samples = train_images.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "classical-insulin",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "60000"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "n_samples"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "younger-lighting",
   "metadata": {},
   "source": [
    "**Create variable for dataset**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "sudden-hearing",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, y_train = datasets.make_classification(n_samples=n_samples, n_features=n_dims)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "following-millennium",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(60000, 144)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aggressive-details",
   "metadata": {},
   "source": [
    "**Get HOG feature from each image & put into dataset variable**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "handy-spanking",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(n_samples):\n",
    "    X_train[i], _ = hog(train_images[i].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')\n",
    "    y_train[i] = train_labels[i]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "present-exhibit",
   "metadata": {},
   "source": [
    "## 3. Training with SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "89e46b33",
   "metadata": {},
   "outputs": [],
   "source": [
    "svm = SVC()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "ec817e6a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>SVC()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SVC</label><div class=\"sk-toggleable__content\"><pre>SVC()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "SVC()"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "svm.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "described-intersection",
   "metadata": {},
   "source": [
    "## 4. Predict Test Dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "remarkable-sample",
   "metadata": {},
   "outputs": [],
   "source": [
    "n_samples = test_images.shape[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "concrete-forth",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test, y_test = datasets.make_classification(n_samples=n_samples, n_features=n_dims)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "naked-credits",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in range(n_samples):\n",
    "    X_test[i], _ = hog(test_images[i].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2')\n",
    "    y_test[i] = test_labels[i]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "dbe3919a",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = svm.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "multiple-hayes",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x13793afd0>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(test_images[20].reshape(28,28), cmap='gray')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "neural-patrick",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "9\n"
     ]
    }
   ],
   "source": [
    "prediksi = svm.predict(X_test[20].reshape(1, n_dims))\n",
    "hasil_prediksi = prediksi[0]\n",
    "print(hasil_prediksi)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "suburban-backing",
   "metadata": {},
   "source": [
    "## 5. Evaluation Metrics"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "54b21123",
   "metadata": {},
   "source": [
    "5.1 Conf matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "friendly-ceiling",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "shaped-lawyer",
   "metadata": {},
   "outputs": [],
   "source": [
    "conf_mat = confusion_matrix(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "failing-edmonton",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 978,    0,    1,    0,    0,    0,    1,    0,    0,    0],\n",
       "       [   0, 1127,    2,    0,    3,    0,    1,    0,    2,    0],\n",
       "       [   3,    0, 1009,    5,    3,    0,    0,    8,    2,    2],\n",
       "       [   2,    0,    7,  977,    0,    5,    0,    6,    7,    6],\n",
       "       [   5,    2,    0,    1,  952,    0,    4,    4,    1,   13],\n",
       "       [   2,    0,    1,    9,    1,  869,    2,    2,    5,    1],\n",
       "       [   4,    2,    1,    0,    3,    5,  942,    0,    1,    0],\n",
       "       [   0,    3,   13,    3,   10,    2,    0,  985,    2,   10],\n",
       "       [   3,    3,    2,    6,    2,    2,    0,    1,  949,    6],\n",
       "       [   2,    0,    0,    5,   11,    2,    0,    5,   13,  971]])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "conf_mat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "representative-prerequisite",
   "metadata": {},
   "outputs": [],
   "source": [
    "from mlxtend.plotting import plot_confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "flexible-savannah",
   "metadata": {},
   "outputs": [],
   "source": [
    "class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "coupled-apache",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, ax = plot_confusion_matrix(conf_mat=conf_mat, class_names=class_names)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "unnecessary-responsibility",
   "metadata": {},
   "source": [
    "### 5.2 Precission"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "unauthorized-queue",
   "metadata": {},
   "source": [
    "$$\\text { Precision }=\\frac{T P}{T P+F P}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "protected-suggestion",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import precision_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "centered-sector",
   "metadata": {},
   "outputs": [],
   "source": [
    "precision = precision_score(y_test, y_pred, average=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "collect-status",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.97897898, 0.99120493, 0.97393822, 0.97117296, 0.96649746,\n",
       "       0.9819209 , 0.99157895, 0.97428289, 0.96639511, 0.96233895])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "serious-tiger",
   "metadata": {},
   "source": [
    "### 5.3 Recall"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "weekly-andrews",
   "metadata": {},
   "source": [
    "$$\\text { Recall }=\\frac{T P}{T P+F N}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "linear-strap",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import recall_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "canadian-valuable",
   "metadata": {},
   "outputs": [],
   "source": [
    "recall = recall_score(y_test, y_pred, average=None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "confident-brave",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.99795918, 0.99295154, 0.97771318, 0.96732673, 0.9694501 ,\n",
       "       0.97421525, 0.98329854, 0.95817121, 0.97433265, 0.96233895])"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recall"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "small-aircraft",
   "metadata": {},
   "source": [
    "### 5.4 Accuracy"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "checked-assembly",
   "metadata": {},
   "source": [
    "$$\\text { Accuracy }=\\frac{T P+T N}{T P+T N+F P+F N}$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "ranking-silence",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "posted-worst",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy = accuracy_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "joined-listening",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9759"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "sublime-barrel",
   "metadata": {},
   "source": [
    "### 5.5 F1-Score"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "after-prediction",
   "metadata": {},
   "source": [
    "$$\\text { F1-Score }=\\left(\\frac{2}{\\text { precision }^{-1}+\\text { recall }^{-1}}\\right)=2 \\cdot\\left(\\frac{\\text { precision } \\cdot \\text { recall }}{\\text { precision }+\\text { recall }}\\right)$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "bizarre-bermuda",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "hungarian-probe",
   "metadata": {},
   "outputs": [],
   "source": [
    "f1 = f1_score(y_test, y_pred, average='macro')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "floating-corner",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9757815807547701"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
