
# coding: utf-8

# # CS155: Miniproject 1
# Kavya Sreedhar, Audrey Wang, Anne Zhou

# In[1]:


import numpy as np 
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDRegressor, Ridge, ARDRegression
from sklearn.model_selection import cross_val_score


# In[18]:


# Seed the random number generator.
np.random.seed(1)

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


# In[4]:


# Load data.
train = load_data('training_data.txt')
X_test = load_data('test_data.txt')

X_train = train[:, 1:]
y_train = train[:, 0]
N_train = len(X_train)
N_test = len(X_test)


# In[19]:


# Normalize data.
max_vals = X_train.max(axis=0)
X_train = X_train / max_vals
X_test = X_test / max_vals


# In[21]:


# Find cross-validation score of different models to determine best one.
X_val = X_train[:4000]
y_val = y_train[:4000]
#types = ['svm', 'logistic regression', 'random forest', 'gradient boost']
#notes: also tried SGD Regressor. nit was [rettu awfi;]
#types = ['logistic regression', 'Linear SVC with hinge', 'Linear SVC with squared hinge', 'Ridge']
types = ['AdaBoostClassifier']
scores = []

sv = svm.SVC()
log = LogisticRegression()
random_forest = RandomForestClassifier()
gradient_boost = GradientBoostingClassifier()
hinge = LinearSVC(loss='hinge')
squared_hinge = LinearSVC(loss = 'squared_hinge')
ridge = Ridge()
#scores.append(cross_val_score(sv, X_val, y_val))
#scores.append(cross_val_score(log, X_train, y_train))
#scores.append(cross_val_score(random_forest, X_val, y_val))
#scores.append(cross_val_score(gradient_boost, X_val, y_val))
#scores.append(cross_val_score(hinge, X_train, y_train))
#scores.append(cross_val_score(squared_hinge, X_train, y_train))
#scores.append(cross_val_score(ridge, X_train, y_train))
scores.append(cross_val_score(AdaBoostClassifier(n_estimators = 1000), X_train, y_train))
for i in range(len(types)):
    print('%s: %f' % (types[i], np.mean(scores[i])))
    print(scores[i])
    print()

'''
# In[29]:
#tried all C_vals in both lists, noticed 0.05 did best, so tried closer values
# in second C_vals list
C_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0]
#C_vals = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
means = []
for C in C_vals:
    print("C: ", C)
    l2hinge = LinearSVC(loss='hinge', penalty='l2', C = C)
    score1 = cross_val_score(l2hinge, X_train, y_train)
    print("hinge: ", score1) 
    print("Mean: ", np.mean(score1))
    l2shinge = LinearSVC(loss='squared_hinge', penalty='l2', C = C)
    score2 = cross_val_score(l2shinge, X_train, y_train)
    print("squared_hinge: ", score2)
    print("Mean: ", np.mean(score2))
    means.append(np.mean(score1))
    means.append(np.mean(score2))

print("Hinge Max: ", max(means))
'''

# Tweak parameters for best classifiers.
c_arr = [0.3, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0]
#c_arr = [1.0, 1.3, 1.5, 1.7, 2.0, 2.2, 2.5, 2.7, 3.0]
means = []
log1 = LogisticRegression(class_weight='balanced')
score1 = cross_val_score(log1, X_train, y_train)
print('only balanced: %f' % np.mean(score1))
print(score1)

for i in range(len(c_arr)):
    print(c_arr[i])

    '''
    log2 = LogisticRegression(C=c_arr[i])
    score2 = cross_val_score(log2, X_train, y_train)
    print('only c: %f' % np.mean(score2))
    print(score2)

    log3 = LogisticRegression(C=c_arr[i], solver='sag')
    score3 = cross_val_score(log3, X_train, y_train)
    print('sag: %f' % np.mean(score3))
    print(score3)

    log4 = LogisticRegression(C=c_arr[i], class_weight='balanced')
    score4 = cross_val_score(log4, X_train, y_train)
    print('both c and balanced: %f' % np.mean(score4))
    print(score4)

    log5 = LogisticRegression(C=c_arr[i], class_weight='balanced', solver='newton-cg')
    score5 = cross_val_score(log5, X_train, y_train)
    print('balanced and newton : %f' % np.mean(score5))
    print(score5)

    log6 = LogisticRegression(C=c_arr[i], solver='newton-cg')
    score6 = cross_val_score(log6, X_train, y_train)
    print('newton %f' % np.mean(score6))
    print(score6)
    print()

    log5 = LogisticRegression(C=c_arr[i], solver='sag', class_weight='balanced')
    score5 = cross_val_score(log5, X_train, y_train)
    print('sag and balanced: %f' % np.mean(score5))
    print(score5)
    print()

    means.append(np.mean(score2))
    means.append(np.mean(score3))
    means.append(np.mean(score4))
    means.append(np.mean(score5))
    
    means.append(np.mean(score4))
    means.append(np.mean(score5))
    means.append(np.mean(score6))
    

print(max(means))
'''
# In[30]:

#0.05 was minimum from first C_vals, 0.06 was minimum from second C_vals
loss = 'squared_hinge'
C = 0.06
# Make predictions to output file.
#clf = LogisticRegression(C=2.0, class_weight='balanced')
clf = LinearSVC(loss = loss, penalty='l2', C = C)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test).flatten()
f = open('predictions.txt', 'w')
f.write('Id,Prediction\n')
for i in range(len(predictions)):
    f.write('%d,%d\n' % ((i + 1), predictions[i]))
f.close()

# In[ ]:


# import tensorflow as tf 
# import keras
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation, Flatten, Dropout
# from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
# from keras import regularizers


# In[5]:


# N = 1000 # Number of parameters

# # Define the model.
# model = Sequential()
# model.add(Dense(1000, input_shape=(N,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))

# model.add(Dense(900))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Dense(800))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))

# model.add(Dense(10))
# model.add(Dense(1))

# # Print number of params
# model.count_params()

# # Compile the model
# model.compile(optimizer='adam',
#               loss='mse',
#               metrics=['accuracy'])

# # Train the model for 1 epoch
# history = model.fit(X_train, y_train, epochs=1, batch_size=32)

# # Evaluate the model
# model.evaluate(x=X_train, y=y_train)


# In[28]:




