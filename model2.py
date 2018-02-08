
# coding: utf-8

# # CS155: Miniproject 1
# Kavya Sreedhar, Audrey Wang, Anne Zhou
import numpy as np 
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDRegressor, Ridge, ARDRegression
from sklearn.linear_model import TheilSenRegressor, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier

# Define function for loading files
def load_data(filename, skiprows=1):
    """
    Function loads data stored in the file filename and returns it as a numpy ndarray.
    
    Inputs:
        filename: given as a string.
        
    Outputs:
        Data contained in the file, returned as a numpy ndarray
    """
    return np.loadtxt(filename, skiprows=skiprows, delimiter=' ')

# Load data.
train = load_data('training_data.txt')
X_test = load_data('test_data.txt')

X_train = train[:, 1:]
y_train = train[:, 0]
N_train = len(X_train)
N_test = len(X_test)

# Normalize data.
max_vals = X_train.max(axis=0)
X_train = X_train / max_vals
X_test = X_test / max_vals

# Find cross-validation score of different models to determine best one.
X_val = X_train[:5000]
y_val = y_train[:5000]

one = LogisticRegression(C=2.0, class_weight='balanced', solver = 'newton-cg')
two = LogisticRegression(C=2.2, class_weight='balanced', fit_intercept=False, solver =  'newton-cg')
three = RandomForestClassifier(max_depth=150, min_samples_split=50, n_estimators=75)
#four = LogisticRegression(C=0.8) -> 0.8485
four = LogisticRegression(C=0.8, solver = 'newton-cg', class_weight='balanced')
five = SGDClassifier(loss='log', alpha=0.0005)

# five = LogisticRegression(C=1.4, class_weight='balanced')
clf = VotingClassifier(estimators=[('lr1', one), ('lr2', two), ('rf', three), ('four', four), ('five', five)], 
                           voting='soft')

score = cross_val_score(clf, X_train, y_train)
print('voting classifier:', np.mean(score))
print(score)

# Make predictions to output file.
clf.fit(X_train, y_train)
predictions = clf.predict(X_test).flatten()
f = open('predictions.txt', 'w')
f.write('Id,Prediction\n')
for i in range(len(predictions)):
    f.write('%d,%d\n' % ((i + 1), predictions[i]))
f.close()