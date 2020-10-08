""" generates near orthogonal MC samples in opt-NOMC """

import numpy as np
import timeit
import json
import os


''' gram schmidt decomposition '''
def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q





''' initialize NOMC with several normalized independent orthogonal blocks '''
def ortho_starting(d, N):
    turns = int(N/d)
    remainder = N%d
    
    V =  np.random.normal(size=[(turns+1)*d, d])
    V /= np.linalg.norm(V, axis=1).reshape([(turns+1)*d,1])

    V_ = np.zeros_like(V)
    
    for i in range(turns+1):
        v = gram_schmidt_columns(V[i*d:(i+1)*d, :])
        V_[i*d:(i+1)*d, :] = v
    return V_[:-(d-remainder),:]



''' initialize NOMC with several independent orthogonal blocks '''
def ortho_starting2(d, N):
    turns = int(N/d)
    remainder = N%d
    
    V = np.random.normal(size=[(turns+1)*d, d]) 
    V_norm = np.linalg.norm(V, axis=1)
    V_norm = V_norm.reshape(len(V_norm),1) 
    
    V_ = np.zeros_like(V)
    
    for i in range(turns+1):
        v = gram_schmidt_columns(V[i*d:(i+1)*d, :])
        V_[i*d:(i+1)*d, :] = v
    
    return V_[:-(d-remainder),:] * ( V_norm[:-(d-remainder)] )





def opt_NOMC(params):
    
    """
    Parameters
    -------
        m: INT  
            number of independent batches of sampmles to generate
        d: INT
            dimension of sample vector
        delta & dt: FLOAT
            hyper-parameters in opt-NOMC
        Loop: INT
            number of iterations in opt-NOMC algorithm

    Returns
    -------
    will return m batches of near-orthogonal .npy data files under the output path

    """
    
    batches = params['batches'] # number of independent batches of sampmles to generate
    samples = params['samples']
    d = params['dimension']
    delta = params['delta']
    dt = params['dt']
    Loop = params['Loop']
    path = params['project_path']
    
    
    
    for N in samples:

        outfilepath = path+"/data/d={}/N={}/".format(d, N)
    
        if not os.path.exists(outfilepath):
            os.makedirs(outfilepath)
            
        for m in range(batches):
            
            print("########## d ="+str(d)+ ", N = "+str(N)+", m ="+str(m)+ " ###########")
        
            
            V = ortho_starting2(d, N) # starting from independent orthogonal samples
            V /= np.linalg.norm(V, axis=1).reshape([N,1])
            
            
            t0=timeit.default_timer()
            
            for loop in range(Loop):
                
                F = np.zeros([N, d])
                Energy = np.zeros(N)
                MinDistance = np.inf
                MaxDistance = -np.inf
                
                for n1 in range(N-1):
                    vec1 = V[n1]
                    vec2 = V[n1+1:]
                    
                    f = vec2-vec1
                    test = delta*2 / (delta + np.sum(f**2,1))**2
                    Energy[n1] = sum(delta / (delta + np.sum(f**2,1)))
                    MaxDistance = max(max(np.sum(f**2,1)),MaxDistance)
                    MinDistance = min(min(np.sum(f**2,1)),MinDistance)
                    
                    f = np.multiply(f,test.reshape(len(test),1 ))
                    F[n1] -= np.sum(f,0)
                    F[n1+1:] += f
                    
                    f = vec2+vec1
                    test = delta*2 / (delta + np.sum(f**2,1))**2
                    f = np.multiply(f,test.reshape(len(test),1 ))
                    F[n1] += np.sum(f,0)
                    F[n1+1:] += f
                
                TotalEnergy = sum(Energy)   

                V += dt*F
                V /= np.linalg.norm(V, axis=1).reshape([N,1])
                
                if(loop%1000==0):
                    t1=timeit.default_timer()
                    print( "loop = {}, time passed = {}s, TotalEnergy = {}, MaxDistance = {}, MinDistance = {}".format(loop, round(t1-t0,4), round(TotalEnergy,4), round(MaxDistance,4), round(MinDistance,4)) )
                    t0=timeit.default_timer()

        
            with open(outfilepath + 'sample_d={}_N={}_m={}.npy'.format(d,N,m), 'wb') as f:
                np.save(f, V)
            

    


if __name__ == "__main__":
    
    with open('config_NOMC.json') as json_file:
        params = json.load(json_file)
    
    opt_NOMC(params)
    

    