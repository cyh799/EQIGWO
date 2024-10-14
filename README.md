# EQIGWO
Explorative Binary Grey Wolf Optimizer with Quadratic Interpolation for feature selection

## Introduction

* EQIGWO aims to enhance global search capability and convergence accuracy. The algorithm is based on the Grey Wolf Optimizer (GWO) and incorporates multiple optimization strategies, making it suitable for feature selection optimization tasks.
* The `EQIGWO` provides an example of how to apply EQIGWO on benchmark dataset

## Install Dependencies
* The software is written in Python and supports Python 3.9 and later versions. By using Python 3.9 or higher, the algorithm can achieve optimal performance in the latest Python environment, while also leveraging the newest libraries and tools for expansion and optimization. If you need to extend the program, you can download the required libraries by using the following command:
```code
pip install [library_name]
```

## Usage Instructions
* Load the dataset. 
```code
with open('data/australian.arff', encoding="utf-8") as f:
    header = []
    for line in f:
        if line.startswith("@attribute"):
            header.append(line.split()[1])
        elif line.startswith("@data"):
            break
    df = pd.read_csv(f, header=None)
    df.columns = header
    data  = df.values
    feat  = np.asarray(data[:, 0:-1])
    label = np.asarray(data[:,-1])
```

* Using 10-fold cross-validation to split the dataset".
  This means dividing the dataset into 10 equal subsets, where 9 subsets are used to train the model and 1 subset is used to test the model. This process is repeated 10 times, with a different subset used as the test set each time, ensuring that each subset is used for testing once.
```code
from sklearn.model_selection import  KFold
kfold = KFold(n_splits=10, shuffle=True)
for i,(train_index,test_index) in enumerate(kfold.split(feat)):
    xtrain = feat[train_index]
    xtest = feat[test_index]
    ytrain = label[train_index]
    ytest = label[test_index]
```
* Parameter settings
```code
    k    = 5     # k-value in KNN
    N    = 5    # number of particles
    T    = 30   # maximum number of iterations
```
* Population initialization
```code
    for p in range(N):
        for d in range(dim):
            X[p, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * random.random()
```
* Select an algorithm and optimize it to choose the optimal feature subset
```code
  from Feature_selection import EQIGWO
  feature = EQIGWO(xtrain, ytrain, X)
  x_train   = xtrain[:, feature]
  y_train   = ytrain.reshape(num_train)
  x_valid   = xtest[:, feature ]
  y_valid   = ytest.reshape(num_valid)
```
* Initialize KNN Classifier. Train and test using the selected optimal feature subset
```code
  from sklearn.neighbors import KNeighborsClassifier

# Initialize KNN classifier with a specified number of neighbors (e.g., k)
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(x_train, y_train)
  pred = knn.predict(x_valid)
  accuracy = accuracy_score(y_valid, y_pred)
  Acc = np.sum(y_valid == y_pred)  / num_valid
```
 
