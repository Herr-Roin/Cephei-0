import network
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.e**-x)

def sigdev(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x)

#constants
learn_rate=0.2
mini_batch_size=1 # 1 disables stochastic GD
standard_activation=sigmoid
L1_regularization_constant=0.001
L2_regularization_constant=0.05 #Put 0 to disable L2 Regularization

def backpropagation(data, epochs, overwrite, network,include_validation_test,validation_data, show_plot):
    
    X=data["X"] #X is the input data vector, ai
    Y=data["Y"] #Y is what the output is supposed to be, y
    net=network
    n=0
    A=validation_data["X"]
    B=validation_data["Y"]
    
    #Plotdata and dropout lists
    trainingMSES=[]
    validationMSES=[]
    stats=[]
    valminimum=0
    
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
        
        batch_error = [np.zeros((net.parameters[l+1], 1)) for l in range(len(net.w))]
        batch_weighted_error = [np.zeros((net.parameters[l+1], net.parameters[l])) for l in range(len(net.w))]
        for i in range(len(X)):
            m+=1
            
            z_l=[0]
            a_l=[]
            error=[0 for i in range(1,len(net.parameters))] 
                
            a_l.append(X[i].reshape(-1, 1)) #layer 0
            
            for layer_idx in range(len(net.parameters)-1):
                z_l.append(net.forwardz(a_l[layer_idx], layer_idx+1))

                # Check if this is the LAST layer
                if layer_idx == len(net.parameters)-2:  # Last layer
                    if net.parameters[-1] == 1:
                        a_l.append(sigmoid(z_l[-1]))
                    else:
                        a_l.append(softmax(z_l[-1]))
                else:
                    a_l.append(sigmoid(z_l[-1]))
                    
            if net.parameters[-1] == 1:
                y_target = np.array([[float(Y[i])]])   # shape (1,1)
            else:
                y_target = np.zeros((net.parameters[-1], 1))
                y_target[int(Y[i])] = 1.0

            error[-1] = a_l[-1] - y_target
                        
            for d in reversed(range(len(net.parameters))): #Backpropagating the error
                if d==0:break
                if d==len(net.parameters)-1: continue
                error[d-1]=((net.w[d].transpose()) @ error[d])*sigdev(z_l[d])
            
            for l in range(1,len(net.parameters)): #stochastic gradient descent
             batch_error[l-1]+=error[l-1]
             batch_weighted_error[l-1]+=(error[l-1] @ (a_l[l-1].transpose()))
            if m==mini_batch_size: 
                for f in range(1,len(net.parameters)): # update wheights and biases with stochastic gradient descent
                    net.w[f-1]=(1-(learn_rate*L2_regularization_constant)/len(X))*net.w[f-1] - ((learn_rate*L1_regularization_constant)/len(X))*np.sign(net.w[f-1]) - learn_rate*batch_weighted_error[f-1]/m
                    net.b[f-1]=net.b[f-1] - learn_rate*batch_error[f-1]/m
                    
                batch_error = [np.zeros((net.parameters[l+1], 1)) for l in range(len(net.w))]
                batch_weighted_error = [np.zeros((net.parameters[l+1], net.parameters[l])) for l in range(len(net.w))]
                m=0
                
            if net.parameters[-1] == 1:
                trainingMSE += (a_l[-1].item() - Y[i])**2
            else:
                # One-hot encode for loss calculation
                y_one_hot = np.zeros((net.parameters[-1], 1))
                y_one_hot[int(Y[i])] = 1.0
                trainingMSE += np.mean((a_l[-1] - y_one_hot)**2)
                
        stats.append(([w.copy() for w in net.w],[b.copy() for b in net.b]))
        
        trainingMSE=trainingMSE/len(X)
        trainingMSES.append(float(trainingMSE))
    
        print(f"--Epoch {j+1} end--")
        print("---Epoch Summary---")
        print(f"Training MSE: {trainingMSE}")
            
        if include_validation_test:
            for p in range(len(A)):
                validationMSE+=net.compute_and_calculateMSE_soft(A[p],B[p])
            validationMMSE=(validationMSE.sum()/len(A))
            validationMSES.append(float(validationMMSE.item()))
            print(f"Validation MSE: {validationMMSE}")
        correct=0    
        for i in range(len(A)):
            output = net.computesoftmax(A[i])
            pred = np.argmax(output)
            if pred == B[i]:
                correct += 1
        print(f"Validation Accuracy: {correct/len(A):.2%}")
        print("------------------")
        
    valminimum=validationMSES.index(min(validationMSES))
    net.w = stats[valminimum-1][0]
    net.b = stats[valminimum-1][1]
    print(f"The best MSE at epoch {valminimum} was found to be: {min(validationMSES)}")          
                
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
backpropagation(data,30,False,network.Network([28*28,100,10],False,0),True,valdata,True)
#main.main()