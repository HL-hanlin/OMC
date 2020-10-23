#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tanh Kernel Approximation
"""

import os
import random
import pandas as pd
import math
import datetime
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error 
from scipy.stats import norm

from QMC import halton

def baseline_eval_third(x, G, m):
    """Calculate the result of baseline random feature mapping

        Parameters
        ----------
        x: array, dimension = d
            The data point to input to the baseline mapping
        
        G: matrix, dimension = m*d
            The matrix in the baseline random feature mapping
            
        m: integer
            The number of dimension that we want to reduce to
    """
    return (1/m**0.5) * np.tanh(np.dot(G, x))

def gram_schmidt_columns(X):
    '''
    Using QR decomoposition to obtain orthogonal matrix.
    
    Parameters
    ----------
    X : matrix, dimension = m * d, where m <= d
        Random feature matrix with l2 normalized row.

    Returns
    -------
    Q : matrix, dimension = m * d, where m <= d
        Orthogonal random feature matrix with l2 normalized row.

    '''
    Q, R = np.linalg.qr(X)
    return Q

def orthgonalize(V):
    '''
    Generate matrix with multiple orthogonal blocks

    Parameters
    ----------
    V : matrix, dimension = m * d, where m > d
        Random feature matrix with l2 normalized row.

    Returns
    -------
    V_ : TYPE
        Random feature matrix with l2 normalized row and multiple
        blocks.
    '''
    N = V.shape[0]
    d = V.shape[1]
    turns = int(N/d)
    remainder = N%d
    
    V_ = np.zeros_like(V)
    
    for i in range(turns):
        v = gram_schmidt_columns(V[i*d:(i+1)*d, :].T).T
        V_[i*d:(i+1)*d, :] = v
    if remainder != 0:
        V_[turns*d+1:,:] = gram_schmidt_columns(V[turns*d+1:,:].T).T
        
    return V_

def find_sigma(random_sample):
    '''
    Find an appropriate scaling parameter for the kernel value.

    Parameters
    ----------
    random_sample : list
        Store some samples from the dataset.

    Returns
    -------
    float
        Average of all 50th smallest distances.

    '''
    all_distances = []
    for i in range(len(random_sample)):
        #print(f'Calculating the distance of {i}th samples')
        distances = []
        for j in range(len(random_sample)):
            if j!=i:
                distances.append(np.linalg.norm(random_sample[i] - random_sample[j]))
        distances.sort()
        all_distances.append(distances[50])
    return np.mean(all_distances)

def random_rotation(matrix, G_ortho):
    '''
    Perform random rotation.

    Parameters
    ----------
    matrix : matrix
        The matrix to rotate.
    G_ortho : matrix
        The matrix for .

    Returns
    -------
    result : TYPE
        DESCRIPTION.

    '''
    result = np.zeros_like(matrix)
    m = matrix.shape[0]
    d = matrix.shape[1]
    for i in range(m):
        result[i, :] = np.dot(G_ortho[:d], matrix[i, :])
    return result

def read_in_nomc_matrices(d, d_):
    '''
    Read in pre-calculated near orthogonal random matrices.

    Parameters
    ----------
    d : int
        Dimension of data features.
    d_ : int
        Number of multipliers.

    Returns
    -------
    all_V : list
        A list of NOMC matrices.

    '''
    all_V = []
    for N in [d*i for i in range(1,d_+1)]:
        V_s = []
        for m in range(500):
            try:
                with open(f'{os.getcwd()}/data Apr13/N={N}/V_d=10_N={N}_iter=20000_m={m}.npy', 'rb') as f:
                    # The address above is subject to change
                    V = np.load(f)
                    V_s.append(V)
            except:
                pass
        all_V.append(V_s) 
        
    return all_V

def generate_data(d):
    '''
    Data generator. Generate data for kernel approximation    

    Parameters
    ----------
    d : int
        Dimension of data features.

    Returns
    -------
    data : list
        A list of features.
    '''
    
    letters = pd.read_csv('letter-recognition.csv')
    letters = np.asarray(letters)
    data = np.zeros((letters.shape[0], d))
    for i in range(letters.shape[0]):
        for j in range(1, data.shape[1]+1):
            data[i,j-1] = letters[i,j]
    data = list(data)
    
    return data

def generate_halton_sequences(episode, epoch):
    '''
    Generate halton sequences
    
    Parameters
    ----------
    episode: int
        Number of outer experiments.
    epoch: int
        Number of inner experiments.

    Returns
    -------
    all_halton : list
        A list of halton sequences.

    '''
    
    all_halton = []
    for i in tqdm(range(1, epoch * episode + 1), position = 0, leave = True):
        all_halton.append(halton(i, 10))

    return all_halton
    
    

def generate_qmc_features(d, d_, epoch, all_halton):
    '''
    Generate random features using Quasi Monte Carlo, which leverages
    the halton sequences

    Parameters
    ----------
    d : int
        Dimension of data features.
    d_ : int
        Number of multipliers.
    epoch : int
        Number of inner experiments.
    all_halton : list
        A list of halton sequences.

    Returns
    -------
    None.

    '''
    all_V_qmc = [] 
    for n in [d*i for i in range(1,d_+1)]:
        V_qmc = []
        for i in range(epoch):
            V_qmc.append(norm.ppf(all_halton[n*i: n*(i+1)]))
        all_V_qmc.append(V_qmc)
    
    return all_V_qmc
        
def plot(episode, d_, MSE_iid_, MSE_qmc_, MSE_orthog_, MSE_max_ang_):
    '''
    Plot the MSE at each multiplier of all methods.

    Parameters
    ----------
    episode : int
        Number of outer experiments.
    d_ : int
        Number of multipliers.
    MSE_iid_ : list
        MSEs of MC in all experiments.
    MSE_qmc_ : list
        MSEs of QMC in all experiments.
    MSE_orthog_ : list
        MSEs of BOMC in all experiments.
    MSE_max_ang_ : list
        MSEs of NOMC in all experiments.

    Returns
    -------
    None.

    '''
    x1 = range(1, d_+1)
    x1 = x1 + np.zeros((episode, d_))
    x1 = np.sort(x1.reshape(episode*d_,)) 
    
    mse_iid_ = np.asarray(MSE_iid_).reshape(-1)
    category = ['iid' for i in range(episode*d_)]
    df1 = pd.DataFrame({'D/d': x1, 'MSE':mse_iid_, 'Category':category})
    
    mse_qmc_ = np.asarray(MSE_qmc_).reshape(-1)
    category = ['QMC' for i in range(episode*d_)]
    df2 = pd.DataFrame({'D/d': x1, 'MSE':mse_qmc_, 'Category':category})
    
    mse_ortho_ = np.asarray(MSE_orthog_).reshape(-1)
    category = ['OG' for i in range(episode*d_)]
    df3 = pd.DataFrame({'D/d': x1, 'MSE':mse_ortho_, 'Category':category})
    
    mse_max_ang_ = np.asarray(MSE_max_ang_).reshape(-1)
    category = ['Particle' for i in range(episode*d_)]
    df4 = pd.DataFrame({'D/d': x1, 'MSE':mse_max_ang_, 'Category':category})
    
    df = pd.concat([df1, df2, df3, df4], ignore_index=True)
    
    
    sns.lineplot(x="D/d", y="MSE", hue="Category",
                       err_style="bars", ci=90, data=df)
    plt.title('Tanh Kernel Approximation')
    #plt.savefig(f'Gaussian_kernel_Comparison(with iid)_{datetime.datetime.now()}.png', dpi = 600)
    plt.show()

def main(d, d_):
    
    #d = 10  # dimension of the data point 
    N = [d*i for i in range(1,d_+1)]  # number of samplings
    episode = 100
    epoch =  450# number of experiments to perform for each episode

    data = generate_data(d)  # read in test data
    
    all_halton = generate_halton_sequences(episode, epoch)
    all_V_qmc = generate_qmc_features(d, d_, epoch, all_halton)
    all_V = read_in_nomc_matrices(d, d_)  # read in pregenerated nomc matrices
    
    iid_estimates = np.zeros((d_, episode, epoch))
    qmc_estimates = np.zeros((d_, episode, epoch))
    orthog_estimates = np.zeros((d_, episode, epoch))
    max_ang_estimates = np.zeros((d_, episode, epoch))
            
    MSE_iid = []
    MSE_qmc = []
    MSE_orthog = []
    MSE_max_ang = []
    
    MSE_iid_ = []
    MSE_qmc_ = []
    MSE_orthog_ = []
    MSE_max_ang_ = []
    
    list_of_samples = []
    true_values = []
    
    random_sample = random.sample(data, 1000)
    sigma = find_sigma(random_sample)
    
    random_sample = [i/(2*sigma) for i in random_sample]
    
    m = 10*5000
    
    for e in tqdm(range(episode), position=0, leave=True):
        while True:
            x, y = random.sample(random_sample, 2)
            if np.all(x == np.zeros(d))==0 and np.all(y == np.zeros(d))==0:
                break
        V = np.random.normal(0, 1, (m,d))   #create N*d iid matrix
        V_orthog = orthgonalize(V)   #create N*d matrix with orthogonal blocks
        norms = np.linalg.norm(V, axis=1).reshape([m,1])
        true_value = np.dot(baseline_eval_third(x, V_orthog*norms, m), 
                            baseline_eval_third(y, V_orthog*norms, m))
        list_of_samples.append((x, y))
        true_values.append(true_value)
    
    sample_V = []
    for i in range(len(N)):
        sample_V.append(random.sample(all_V[i], epoch))
    
    for n in range(len(N)):
        mse_iid_n = []
        mse_qmc_n = []
        mse_orthog_n = []
        mse_max_n = []
        
        for e in tqdm(range(episode), position=0, leave=True):
            #print(f'{N[n]} samplings, {e}th episode')
            true = np.repeat(true_values[e], epoch)
            
            x = list_of_samples[e][0]
            y = list_of_samples[e][1]
            for i in range(epoch):
                np.random.seed()
                V = np.random.normal(0, 1, (N[n],d))   #create N*d iid matrix
                V_orthog = orthgonalize(V)   #create N*d matrix with orthogonal blocks
                norms = np.linalg.norm(V, axis=1).reshape([N[n],1])
                
                iid_estimates[n, e, i] = np.dot(baseline_eval_third(x, V, N[n]),
                            baseline_eval_third(y, V, N[n]))
                qmc_estimates[n, e, i] = np.dot(baseline_eval_third(x, all_V_qmc[n][i], N[n]),
                            baseline_eval_third(y, all_V_qmc[n][i], N[n]))
                orthog_estimates[n, e, i] = np.dot(baseline_eval_third(x, V_orthog*norms, N[n]),
                            baseline_eval_third(y, V_orthog*norms, N[n]))
                
                particle_matrix = random_rotation(sample_V[n][i], V_orthog)
                max_ang_estimates[n, e, i] = np.dot(baseline_eval_third(x, particle_matrix*norms, N[n]),
                            baseline_eval_third(y, particle_matrix*norms, N[n]))
                
            mse_iid_n.append(mean_squared_error(true, iid_estimates[n, e]))
            mse_qmc_n.append(mean_squared_error(true, qmc_estimates[n, e]))
            mse_orthog_n.append(mean_squared_error(true, orthog_estimates[n, e]))
            mse_max_n.append(mean_squared_error(true, max_ang_estimates[n, e]))
        
        MSE_iid_.append(mse_iid_n)
        MSE_iid.append(np.mean(mse_iid_n))
        
        MSE_qmc_.append(mse_qmc_n)
        MSE_qmc.append(np.mean(mse_qmc_n))
        
        MSE_orthog_.append(mse_orthog_n)
        MSE_orthog.append(np.mean(mse_orthog_n))
        
        MSE_max_ang_.append(mse_max_n)
        MSE_max_ang.append(np.mean(mse_max_n))
    
    plot(episode, d_, MSE_iid_, MSE_qmc_, MSE_orthog_, MSE_max_ang_)

if __name__ == "__main__":
    
    d = 10
    d_ = 10
    main(d, d_)

