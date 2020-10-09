""" This code implements sliced Wasserstein distance experiments. 
    We use multivariate gaussian distributions here as an example """


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from opt_NOMC import *
import json





def estimateSWD_iid(  eta_1, eta_2, N, M, p=2):
    
    '''
    M: Number of samples from multivariate normal distributions
    p: p_wasserstein distance. We only use 2 here
    N: Is also the number of samples we need for V on unit sphere
    d: dimension
    '''
    
    d = len(eta_1[0])
    
    SWD = 0

    for n in range(N):
        v = np.random.randn(d)
        v /= np.linalg.norm(v)
        
        proj_eta1 = np.dot(eta_1,v)
        proj_eta2 = np.dot(eta_2,v)
        
        SWD += pow(np.linalg.norm(np.sort(proj_eta1)-np.sort(proj_eta2), p), p) / M
        
    return  pow( float(SWD)/N, 1./p)





def estimateSWD_ort(eta_1, eta_2, N, M, p=2):
    ''' parameters defined same as above '''
    
    d = len(eta_1[0])
    
    SWD = 0

    V = ortho_starting(d, N)
    
    for n in range(N):
        v = V[n,:]
        proj_eta1 = np.dot(eta_1,v)
        proj_eta2 = np.dot(eta_2,v)
        
        SWD += pow(np.linalg.norm(np.sort(proj_eta1)-np.sort(proj_eta2), p), p) / M
        
    return  pow( float(SWD)/N, 1./p)





def estimateSWD_MC(eta_1, eta_2, V, N, M, p=2):

    '''
    M: Number of samples from multivariate normal distributions
    p: p_wasserstein distance. We only use 2 here
    N: Is also the number of samples we need for V on unit sphere
    d: dimension
    V: samples from NOMC or QMC
    '''
    
    SWD = 0
    
    for n in range(N):
        
        v = V[n,:]
        
        proj_eta1 = np.dot(eta_1,v)
        proj_eta2 = np.dot(eta_2,v)
        
        SWD += pow(np.linalg.norm(np.sort(proj_eta1)-np.sort(proj_eta2), p), p) / M
        
    return  pow( float(SWD)/N, 1./p)



    







''' ###################################################### '''
''' ###################################################### '''
''' ###################################################### '''
''' ###################################################### '''





    
def generateRandomPDMatrix(matrixSize): 
    A = np.random.randn(matrixSize,matrixSize)
    B = np.dot(A,A.transpose())
    return B/np.sqrt(matrixSize)




def generate_gaussian_distribution_pair(d, path):
    ''' 
    this function generates two multivariate gaussian distributions to calculate SWD distance later
    we can also generate other types of distributions for SWD applications by creating functions similarly.
    '''

    mu_1 = [0]*d
    mu_2 = [1]*d
 
    sigma_1 = generateRandomPDMatrix(d).tolist()
    sigma_2 = generateRandomPDMatrix(d).tolist()
    
    with open(f'{path}/sigma_1.npy', 'wb') as f:
        np.save(f,sigma_1)
    with open(f'{path}/sigma_2.npy', 'wb') as f:
        np.save(f,sigma_2)
    with open(f'{path}/mu_1.npy', 'wb') as f:
        np.save(f,mu_1)
    with open(f'{path}/mu_2.npy', 'wb') as f:
        np.save(f,mu_2)




def load_distribution_pair(path):
       
    with open(f'{path}/sigma_1.npy', 'rb') as f:
        sigma_1 = np.load(f).tolist()
    with open(f'{path}/sigma_2.npy', 'rb') as f:
        sigma_2 = np.load(f).tolist()
        
    with open(f'{path}/mu_1.npy', 'rb') as f:
        mu_1 = np.load(f).tolist()
    with open(f'{path}/mu_2.npy', 'rb') as f:
        mu_2 = np.load(f).tolist()
        
    return mu_1, mu_2, sigma_1, sigma_2



def load_NOMC_data(project_path):
    all_V = []
    for N in N_list:
        V_s = []
        for m in range(epoch):
            with open(f'{project_path}data/NOMC_precalc/d={d}/N={N}/sample_d={d}_N={N}_m={m}.npy', 'rb') as f:
                V = np.load(f)
            V_s.append(V)
        all_V.append(V_s)
    return all_V



def load_QMC_data(project_path):
    halton_V = []
    for N in N_list:
        V_s = []
        for m in range(epoch):
            with open(f'{project_path}data/QMC_precalc/d={d}/N={N}/sample_d={d}_N={N}_m={m}.npy', 'rb') as f:
                V = np.load(f)
            V_s.append(V)
        halton_V.append(V_s)   
    return halton_V    




def generate_estimations(N_list, epoch, eta_1, eta_2, M, P, V_halton, V_particle):

    EST_iid = np.zeros([len(N_list),epoch])
    EST_ort = np.zeros([len(N_list),epoch])
    EST_particle = np.zeros([len(N_list),epoch])
    EST_halton = np.zeros([len(N_list),epoch])    
    
    for epo in range(epoch):
        for N in N_list:
            
            index = N_list.index(N)
            
            est_iid = estimateSWD_iid(eta_1, eta_2, N,M,p=2)
            EST_iid[index][epo] = est_iid
            
            est_ort = estimateSWD_ort(eta_1, eta_2, N,M,p=2)
            EST_ort[index][epo] = est_ort
        
            V_particle = all_V[index][epo]
            est_particle = estimateSWD_MC(eta_1, eta_2, V_particle, N, M, p=2)
            EST_particle[index][epo] = est_particle
            
            V_halton = halton_V[index][epo]
            est_halton = estimateSWD_MC(eta_1, eta_2, V_halton, N, M, p=2)
            EST_halton[index][epo] = est_halton
            
            print(f" N={N}, epo={epo}, iid={np.round(est_iid,4)}, ort={np.round(est_ort,4)}, halton = {np.round(est_halton,4)}, particle={np.round(est_particle,4)}")
           
    EST_list = [EST_iid, EST_ort, EST_particle, EST_halton]

    with open(f'{SWD_path}/EST_list_epoch={epoch}_distSample={M}.npy', 'wb') as f:
        np.save(f, EST_list)
        
    return EST_list
        


        
        
def plot_MSE(N_list, EST_list, exact_value):
    
    EST_iid = EST_list[0]
    EST_ort = EST_list[1]
    EST_particle = EST_list[2]
    EST_halton = EST_list[3]
    
    MSE_iid = np.zeros(len(N_list))
    MSE_ort = np.zeros(len(N_list))
    MSE_particle = np.zeros(len(N_list))
    MSE_halton = np.zeros(len(N_list))
    
    for i in range(len(N_list)):
        MSE_iid[i] = ( np.sum(np.array((EST_iid[i]) - exact_value)**2) ) / epoch
        MSE_ort[i] = ( np.sum(np.array((EST_ort[i]) - exact_value)**2) ) / epoch
        MSE_particle[i] = ( np.sum(np.array((EST_particle[i]) - exact_value)**2) ) / epoch
        MSE_halton[i] =  ( np.sum(np.array((EST_halton[i]) - exact_value)**2) ) / epoch

    sns.set_style("darkgrid")    
    plt.plot(MSE_iid, '-*', label = 'MSE_iid')
    plt.plot(MSE_halton, '--', label = 'MSE_halton')
    
    plt.plot(MSE_ort, '-^', label = 'MSE_ort')
    plt.plot(MSE_particle, '-v',label = 'MSE_max_ang')
    plt.yscale('log')
    plt.xticks(np.arange(len(N_list)),np.arange(1,len(N_list)))
    plt.legend()
    plt.xlabel('D/d')
    plt.ylabel('MSE_average')
    plt.title('SWD_comparison')
    plt.savefig(f'{SWD_path}SWD_d={d}_maxN={int(N_list[-1]/d)}d.png', dpi = 200)
    plt.show()
    
        
    


    
if __name__ == "__main__":
    
    
    
    with open('config_SWD.json') as json_file:
        params = json.load(json_file)

    project_path = params['project_path']
    SWD_path = params['SWD_path']
    
    d = params['dimension']
    N_list = params['samples']
    epoch = params['batches']
    
    if not os.path.exists(SWD_path):
        os.makedirs(SWD_path)
    
    
    
    '''  generate two multivariate gaussian distributions 
         to calculate the SWD distance later  '''
    
    generate_gaussian_distribution_pair(d, SWD_path)
 
    mu_1, mu_2, sigma_1, sigma_2 = load_distribution_pair(SWD_path)
    
    
    
    
    '''  load NOMC data  '''
    all_V = load_NOMC_data(project_path)

    '''  load QMC data  '''
    halton_V = load_QMC_data(project_path) 
    
    
    
    
    ''' set parameters for distribution  '''

    p = 2 #power of the wasserstein distance
    np.random.seed(9999)

 
    
    
    
    ''' Estimated SWD value '''
    
    M = 100000 #number of samples of the empiric distribution    
    eta_1 = np.random.multivariate_normal(mu_1, sigma_1, M)
    eta_2 = np.random.multivariate_normal(mu_2, sigma_2, M)

    EST_list = generate_estimations(N_list, epoch, eta_1, eta_2, M, p, halton_V, all_V)
    

    '''  use a larger M value to derive the unbiased estimate of true SWD value'''
    M = 500000 #number of samples of the empiric distribution    
    eta_1 = np.random.multivariate_normal(mu_1, sigma_1, M)
    eta_2 = np.random.multivariate_normal(mu_2, sigma_2, M)

    EST_list2 = generate_estimations(N_list, epoch, eta_1, eta_2, M, p, halton_V, all_V)
    
    exact_value = EST_list2[1].mean() # use the mean of iid samples as unbiased estimator for true SWD distance
    
    
    
    
    '''  plot MSE  '''
    
    plot_MSE(N_list, EST_list, exact_value)
    
    
    
    
    

    

    
    
        
    
        
    