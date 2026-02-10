import network
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.e**-x)

def sigdev(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def softmax(x):
    sum=0
    a=np.array([0]*len(x))
    
    for i in range(len(x)):
        sum+=(np.e)**(x[i])
    for s in range(len(x)):
        a[s,0]=((np.e)**(x[s,0]))/sum
    return s
        

#constants
learn_rate=0.3
mini_batch_size=1 # 1 disables stochastic GD
standard_activation=sigmoid
L2_regularization_constant=0.0015 #Put 0 to disable L2 Regularization

def backpropagation(data, epochs, overwrite, network,include_validation_test,validation_data, show_plot):
    
    X=data["X"] #X is the input data vector, ai
    Y=data["Y"] #Y is what the output is supposed to be, y
    net=network
    n=0
    A=validation_data["X"]
    B=validation_data["Y"]
    
    #Plotdata
    trainingMSES=[]
    validationMSES=[]
    trainingACC=[]
    validationACC=[]
    
    for j in range(epochs):
        m=0
        n+=1
        print(f"--Epoch {j+1} intialize--")
        validationMSE=0
        perm = np.random.permutation(len(X)) #shuffle
        X = X[perm]
        Y = Y[perm]
        perm = np.random.permutation(len(A))
        A = A[perm]
        B = B[perm]
        trainingMSE=0
        for i in range(len(X)):
            m+=1
            
            z_l=[0]
            a_l=[]
            error=[0 for i in range(1,len(net.parameters))] 
            if m==1: 
                batch_error=[0 for i in range(1,len(net.parameters))]  
                batch_wheighted_error=[0 for i in range(1,len(net.parameters))] 
                
            a_l.append(X[i].reshape(-1, 1)) #layer 0
            
            for _ in range(len(net.parameters)-1): #Computation using sigmoid or softmax
                z_l.append(net.forwardz(a_l[_],_+1))
                if _==range(len(net.parameters)-1):
                    if len(z_l[-1])==1: a_l.append(standard_activation(z_l[-1]))
                    if len(z_l[-1])>1: a_l.append(softmax(z_l[-1]))
                if _!=range(len(net.parameters)-1): a_l.append(standard_activation(z_l[_+1]))
            
            error[-1]=(a_l[-1]-int(Y[i])) #output error (cross etropy)
            
            for d in reversed(range(len(net.parameters))): #Backpropagating the error
                if d==0:break
                if d==len(net.parameters)-1: continue
                error[d-1]=((net.w[d].transpose()) @ error[d])*sigdev(z_l[d])
            
            for l in range(1,len(net.parameters)): #stochastic gradient descent
             batch_error[l-1]+=error[l-1]
             batch_wheighted_error[l-1]+=(error[l-1] @ (a_l[l-1].transpose()))
            if m==mini_batch_size: 
                for f in range(1,len(net.parameters)): # update wheights and biases with stochastic gradient descent
                    net.w[f-1]=(1-(learn_rate*L2_regularization_constant)/len(X))*net.w[f-1] - learn_rate*batch_wheighted_error[f-1]/m
                    net.b[f-1]=net.b[f-1] - learn_rate*batch_error[f-1]/m
                m=0
                
            trainingMSE+=((a_l[-1].sum()-Y[i].sum())**2)
        
        trainingMSE=trainingMSE/len(X)
        trainingMSES.append(float(trainingMSE))
    
        print(f"--Epoch {j+1} end--")
        print("---Epoch Summary---")
        print(f"Training MSE: {trainingMSE}")
            
        if include_validation_test:
            for p in range(len(A)):
                validationMSE+=net.compute_and_calculateMSE(A[p],B[p])
            validationMMSE=validationMSE/len(A)
            validationMSES.append(float(validationMMSE.item()))
            print(f"Validation MSE: {validationMMSE}")
        print("------------------")
              
                
    if show_plot:
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.plot(list(range(epochs)), trainingMSES)
        ax1.set_title("Training MSE")
        ax2.plot(list(range(epochs)), validationMSES, 'tab:orange')
        ax2.set_title('Validation MSE')
        plt.show()
 
            
        
    if overwrite:
        np.savez(
            "data/params.npz",
            **{f"W{i}": w for i, w in enumerate(net.w)},
            **{f"b{i}": b for i, b in enumerate(net.b)}
                ) 
        


        
data=np.load("data/trainingdata.npz")
valdata=np.load("data/validationdata.npz")
backpropagation(data,35,True,network.Network([28*28,32,1],False,0),True,valdata,True)
#main.main()