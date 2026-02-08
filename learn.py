import network
import numpy as np

#constants
h=0.001
learn_rate=0.2

def sigmoid(x):
    return 1/(1+np.e**-x)

def sigdev(x):
    return (sigmoid(x)*(1-sigmoid(x)))


def backpropagation(data, epochs, overwrite, network,include_validation_test,validation_data):
    
    X=data["X"] #X is the input data vector, ai
    Y=data["Y"] #Y is what the output is supposed to be, y
    net=network
    perm = np.random.permutation(len(X)) #shuffle
    X = X[perm]
    Y = Y[perm]
    
    n=0
    lis=[]
    A=validation_data["X"]
    B=validation_data["Y"]
    perm = np.random.permutation(len(A))
    A = A[perm]
    B = B[perm]
    for j in range(epochs):
        n+=1
        print(f"--Epoch {j+1} intialize--")
        validationMSE=0
        perm = np.random.permutation(len(X)) #shuffle
        X = X[perm]
        Y = Y[perm]
        perm = np.random.permutation(len(A))
        A = A[perm]
        B = B[perm]
        for i in range(len(X)):
            trainingMSE=0
            z_l=[0]
            a_l=[]
            error=[0 for i in range(1,len(net.parameters))]
            a_l.append(X[i].reshape(-1, 1)) #layer 0
            
            for _ in range(len(net.parameters)-1):
                z_l.append(net.forwardz(a_l[_],_+1))
                a_l.append(sigmoid(z_l[_+1]))
            
            error[-1]=(a_l[-1]-int(Y[i]))*sigdev(z_l[-1]) #output error
            
            for d in reversed(range(len(net.parameters))): #Backpropagating the error
                if d==0:break
                if d==len(net.parameters)-1: continue
                error[d-1]=((net.w[d].transpose()) @ error[d])*sigdev(z_l[d])
                
                
            for f in range(1,len(net.parameters)): # update wheights and biases
                net.w[f-1]=net.w[f-1] - learn_rate*(error[f-1] @ (a_l[f-1].transpose()))
                net.b[f-1]=net.b[f-1] - learn_rate*(error[f-1])

            trainingMSE+=((a_l[-1].sum()-Y[i].sum())**2)/len(a_l[-1])
            
        print(f"--Epoch {j+1} end--")
        print("---Epoch Summary---")
        print(f"Training MSE: {trainingMSE}")
            
        if include_validation_test:
            for p in range(len(A)):
                validationMSE+=net.compute_and_calculateMSE(A[p],B[p])
            validationMMSE=validationMSE/len(A)
            print(f"Validation MSE: {validationMMSE}")
            if n==5: 
                n=0
        print("------------------")
                    
                
            
            
        
    if overwrite:
        np.savez(
            "data/params.npz",
            **{f"W{i}": w for i, w in enumerate(net.w)},
            **{f"b{i}": b for i, b in enumerate(net.b)}
                ) 
        

        
        
data=np.load("data/trainingdata.npz")
valdata=np.load("data/validationdata.npz")
backpropagation(data,100,True,network.Network([28*28,32,1],False,0),True,valdata)
#main.main()