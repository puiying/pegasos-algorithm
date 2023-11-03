import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# define function to process file
def process_file(filename, process_label=0):
    
    # read csv file
    data = np.genfromtxt(filename, delimiter=',')

    # split data into X and y
    X = data[:,1:]
    y = data[:,0].astype(int)

    # standardize X
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)

    # process_label is true
    if process_label==1:
        y[y==np.unique(y)[0]] = 1
        y[y!=np.unique(y)[0]] = -1

    return (X, y)

# define funciton to randomly select data for the minibatch update
def generateMiniBatchIndex(c1Index, cm1Index, k):
    dataLen = len(c1Index) + len(cm1Index)
    if k != 1/dataLen:
        
        # identify number of datapoints we need for each label based on k
        lenC1 = int(k * len(c1Index))
        lenCM1 = int(k * len(cm1Index))
        
        # randomly choose k% data points from each group
        ranC1Index = np.random.choice(c1Index, size = lenC1)
        ranC2Index = np.random.choice(cm1Index, size = lenCM1)
        
        # combine both datasets from each group
        return np.hstack((ranC1Index, ranC2Index))
    else:
        return np.random.choice(np.hstack((c1Index, cm1Index)), size = int(np.ceil(k)))

# define predict function to be used in pegasos and softplus
def predict(X, w):
    # Generate predictions based on feature X and weight w
    pred = X.dot(w)
    return pred

# define function to update w in Pegasos algorithm
def pegasos(lamb, a, t, kval, X, y, w):

    # identify indeces that is misclassified
    misindex = predict(X, w) * y < 1

    # reset learning rate eta
    eta = 1/(lamb * (t+1))

    # Calculate w_1/2 based on misclassifications, eta, lambda, and k
    w_h = (1-eta * lamb) * w + eta/kval * y[misindex].dot(X[misindex])

    # Update weight w
    w = min(1, (1/np.sqrt(lamb))/(np.linalg.norm(w_h)+1e-15)) * w_h

    # Define loss function
    p = hinge_loss
    
    return w, p

# define simoid function
def sigmoid(x):
    
    # implement the sigmoid function
    return 1/(1+np.exp(-x))

# define gadient update for softplus function to update w
def softplus(lamb, a, t, kval, X, y, w):

    # calculate gradient
    grad = (sigmoid((1 - y * (X @ w)) / a) * -y).T @ X / len(y) + 2 * lamb * w
    
    # update w, assume learning rate = lamb
    w -= lamb * grad
    
    # define loss function
    p = softplus_loss

    return w, p

# calculate hinge loss
def hinge_loss(X, y, w, a):
    l = 1/len(X) * ((predict(X, w)*y)<1).T @ (1 - predict(X, w)*y)   
    return l

# calculate hinge loss with softplus
def softplus_loss (X, y, w, a):
    l = a * np.log(1 + np.exp((1 - y * X.dot(w)) / a))
    return l.mean(axis=0)

# define the main function of stochastic gradient descent
def sgd(filename, k, numruns, gd_function_):

    # Process file and split into X and y
    X, y = process_file(filename, 1)

    # Extract X dimension and indexes
    n, d = X.shape
    index = np.arange(n)

    # Add bias term to X
    X = np.c_[np.ones(n), X]

    # Determine ktot, termination condition
    ktot = 100 * n

    # Initialize lambda and a
    l = 0.001
    a = 0.05

    # Indices of each class
    c1_index = np.where(y == 1)[0]
    cminus1_index = np.where(y == -1)[0]

    # put placeholder for run time
    run_time = pd.DataFrame(columns=['kval', 'numruns', 'runtime'])

    # create plots
    fig, axs = plt.subplots(len(k), numruns, sharex=True, sharey=True, figsize=(15,15))
    
    # For each run(different k values)
    for i in range(len(k)):
        for j in range(numruns):
            
            # Record start time each k'th run
            start_time = time.time()
            # put placeholder for primal objective
            primal_obj = np.zeros((ktot,1))
            # K values (0.0005, 0.01, 0.1, 0.5, 1)
            k_val = k[i]
            # Initialize weights, w, multiply by 0.01 to make w smaller
            w = np.random.rand(d+1) * 0.01

            # Enter gradient descent until termination condition
            for kt in range(ktot):
                # Randomly choose k sample
                k_index = generateMiniBatchIndex(c1_index, cminus1_index, k_val)
                X_k = X[k_index]
                y_k = y[k_index]

                # Updating weight based on the model: pegasos or softplus
                w, p_ = gd_function_(l, a, kt, k_val, X_k, y_k, w)
                
                # Calculate and record primary objective to plot later
                primal_obj[kt] = p_(X, y, w, a) + l * np.linalg.norm(w)**2
            
            # Record end time for the run
            end_time = time.time()
            run_time.loc[run_time.shape[0]] = [k_val*100, j, end_time-start_time]
            
            # Plot primal objective function
            axs[i, j].plot(primal_obj)
            axs[i, j].set_title('k='+'{:.2%}'.format(k_val)+' run #'+str(j))

    # Calculate average and standard diviation of run time
    print('The average and standard deviation of run time are:')
    display(run_time.groupby(['kval'], as_index=False).agg({'runtime':['mean','std']}))
    
    # Show plot
    print('The primal objective over iterations for each k and each run:')
    plt.show()
    
    return ()
