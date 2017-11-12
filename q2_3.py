'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def obs_in_class(data,labels,k):
     class_idx = np.where(labels == k)
     class_labels = labels[class_idx]
     class_data = data[class_idx,][0]
     return(class_labels,class_data)

def compute_parameters(train_data, train_labels):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    train data is a n x 64 numpy array of observations
    train labels is a n x 1 numpy array of their class labels
    '''
    eta = np.zeros((10, 64))
    #train on one class at a time
    for k in np.unique(train_labels):
        k = int(k)
        # get observations in each class and their labels
        lab_class, data_class = obs_in_class(train_data,train_labels,k)
        #add one row of positive cases and one row of negative cases as the prior on eta
        data_class = np.append(data_class,np.zeros((1,64)),axis=0)
        data_class = np.append(data_class,np.ones((1,64)),axis=0)
        #compute the sum of positive values for each feature in class k
        lab_sum = np.sum(data_class,axis=0)
        class_obs = data_class.shape[0]
        eta[k-1,:] = lab_sum/class_obs
    return eta

def plot_images(class_images,filename):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    images is a 10 x 8 x 8 numpy array
    '''
    img_all = np.zeros((10,8,8))
    for i in range(10):
        img_i = class_images[i]
        img_i = img_i.reshape(8,8)
        img_all[i] = img_i
    allconcat = np.concatenate(img_all,1)
    plt.figure(figsize=(20, 5))
    plt.imshow(allconcat,cmap='gray')
    plt.savefig(filename)
        # ...

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

   eta is a 10 x 64 numpy array giving the chance of a positive trial for each feature in each class from compute_parameters
    '''
    generated_data = np.zeros((10, 64))
    for k in range(10):
        new_point = np.zeros((1,64))
        for d in range(eta.shape[1]):
            #sample one point from a binomial with probability eta
            new_point[0,d] = np.random.binomial(1,eta[k,d])
        generated_data[k,:] = new_point
    #plot the resulting images
    plot_images(generated_data,'q2_3_new.png')

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    bin_digits is a n x 64 numpy array of binarized features
    eta is a 10 x 64 numpy array giving the chance of a positive trial for each feature in each class from compute_parameters

    '''
    prob_samp = np.zeros((bin_digits.shape[0],10))
  #  iterate over samples
    for i in range(bin_digits.shape[0]):
        prob_k = np.zeros((1,10))
   #   iterate over classes
        for k in range(10):
            t1_k = np.zeros((64))
            t2_k = np.zeros((64))
            #iterate over each feature
            for d in range(64):
                bin_d = bin_digits[i,d]
                eta_d = eta[k,d]
                t1_k[d] = bin_d*np.log(eta_d/(1-eta_d))
                t2_k[d] = np.log(1-eta_d)
            #sum over features and then sum terms for each class
            t1_k_sum = np.sum(t1_k)
            t2_k_sum = np.sum(t2_k)
            prob_k[0,k] = t1_k_sum + t2_k_sum
        prob_samp[i,:] = prob_k
  
    return prob_samp

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    bin_digits is a n x 64 numpy array of binarized features
    eta is a 10 x 64 numpy array giving the chance of a positive trial for each feature in each class from compute_parameters

    '''
    cond_lik = np.zeros((bin_digits.shape[0],10))
    gen_lik = generative_likelihood(bin_digits,eta)
    prior = np.log(1/10)
    #sum the likelihoods for all classes times the prior to compute the evidence
    evidence = np.sum(np.exp(gen_lik)*np.exp(prior),axis=1)
    evidence_mat = np.repeat(evidence,10,axis=0)
    evidence_mat = evidence_mat.reshape(bin_digits.shape[0],10)
    out = (gen_lik +prior) - np.log(evidence_mat)
 
    return out

def right_class(x,k):
    k = int(k)
    if k == 0:
        k = 10
    return(x[k-1])

def avg_conditional_likelihood(bin_digits, eta, labels):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    bin_digits is a n x 64 numpy array of binarized features
    eta is a 10 x 64 numpy array giving the chance of a positive trial for each feature in each class from compute_parameters
    labels is an n x 1 vector of labels for observations in bin_digits
    '''
    #obtain posterior likelihoods for each observation
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    right_cond_lik = np.zeros((bin_digits.shape[0]))
    #only sum the posterior likelihoods from the correct class
    for i in range(bin_digits.shape[0]):
        right_cond_lik[i] = right_class(cond_likelihood[i,:],labels[i])
    #average and return
    right_cond_sum = np.sum(right_cond_lik)
    right_cond_mean = right_cond_sum/bin_digits.shape[0]
    return right_cond_mean

def best_class (x):
#pick out the class with the highest posterior likelihood
    best_k = np.argmax(x)+1
    if (best_k == 10):
        best_k = 0
    return(best_k)

def classification_accuracy(predicted_labels,eval_labels):
    '''
    computes accuracy given a set of prediced labels and truth labels
    predicted_labels is an n x 1 vector of predictions
    eval_labels is an n x 1 vector of true labels against which to evaluate the predictors
    '''
    # Combine predicted and evaluation into a single array
    pred_truth = np.zeros(predicted_labels.shape[0],dtype={'names':('predicted','truth'),'formats':('int8','int8')})
    pred_truth['predicted'] = predicted_labels
    pred_truth['truth']= eval_labels
    #count the rows where the predicted matches the truth
    correct = pred_truth[pred_truth['predicted'] == pred_truth['truth']]
    return(correct.shape[0]/pred_truth.shape[0])

def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    eta is a 10 x 64 numpy array giving the chance of a positive trial for each feature in each class from compute_parameters
means is a 10 x 64 numpy array
    bin_digits is a n by 64 numpy array
    '''
    # Compute posterior likelihood and return the most likely class
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    pred = np.apply_along_axis(best_class, 1 , cond_likelihood)
   
    return(pred)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    plot_images(eta,'q2_3_eta.png')
   #generate new data
    generate_new_data(eta)
   # Evaluate model
    avg_train = avg_conditional_likelihood(train_data,eta,train_labels)
    print("average training  conditional log likelihood", avg_train, "average training conditional likelihood", np.exp(avg_train) )
    avg_test = avg_conditional_likelihood(test_data,eta,test_labels)
    print("average testing conditional log likelihood", avg_test, "average testing conditional likelihood", np.exp(avg_test))
    pred_train = classify_data(train_data,eta)
    acc_train = classification_accuracy(pred_train,train_labels)
    pred_test = classify_data(test_data,eta)
    acc_test = classification_accuracy(pred_test,test_labels)
    print("acc train", acc_train, "acc test", acc_test)
if __name__ == '__main__':
    main()
