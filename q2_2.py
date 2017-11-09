'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def obs_in_class(data,labels,k):
     class_idx = np.where(labels == k)
     class_labels = labels[class_idx]
     class_data = data[class_idx,][0]
     return(class_labels,class_data)
 
def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    for k in np.unique(train_labels):
        k= int(k)
        class_labels , class_data = obs_in_class(train_data,train_labels,k)
        sum_k = np.sum(class_data,axis=0)
        avg_k = sum_k/class_data.shape[0]
        means[0,:] = avg_k
        #   means=np.insert(means,0,avg_k,axis=0)
        print(avg_k)
        print(means[(k-1),:])

    # Compute means
    return means

def compute_sigma_mles(train_data, train_labels, means):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    for k in np.unique(train_labels):
        print("k: ", k)
        k=int(k)
        class_labels , class_data = obs_in_class(train_data,train_labels,k)
        print(class_labels.shape)
        print(class_data.shape)
        obs_diff = np.zeros((class_data.shape)).transpose()
        print(obs_diff.shape)
        class_mean = means[k-1,:]
        print(class_mean.shape)
        for i in np.arange(0,class_data.shape[0]-1):
            obs_data = class_data[i,:].transpose()
            diff = obs_data-class_mean
            obs_diff[:,i]=diff
        print(obs_diff[0:3,])
     #   cov = np.cov(class_data,rowvar=False)
        cov = np.dot(obs_diff,obs_diff.transpose())/class_data.shape[0]
        cov = cov + 0.01
     ##   print(cov.shape)
        covariances[k-1]=cov
            
            
    # Compute covariances
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    cov_mat = np.zeros((10,8,8))
    for i in range(10):
        print(i)
        cov_diag = np.diag(covariances[i])
        print(covariances[i][5:9,5:9])
        cov_diag_log = np.log(cov_diag)
        cov_diag_log = cov_diag_log.reshape(8,8)
        cov_mat[i] = cov_diag_log
    all_concat_cov = np.concatenate(cov_mat, 1)
    plt.imshow(all_concat_cov, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    return None

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    return None

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    return None

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    pass

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
 #   train_labels = train_labels[np.where(train_labels<)]
    # Fit the model
    print(train_data.shape)
    means = compute_mean_mles(train_data, train_labels)
  #  print(means)
    covariances = compute_sigma_mles(train_data, train_labels,means)
    plot_cov_diagonal(covariances)
    # Evaluation

if __name__ == '__main__':
    main()