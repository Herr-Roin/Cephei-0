import numpy as np

def sigmoid(x):
    return 1/(1+np.e**-x)
def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x)

class Network(): #A Neural network specified to have one hidden layer only
    def __init__(self,parameters,use_para,data):
        self.parameters=parameters
        self.layers=len(parameters)
        self.w=[np.array([[np.random.uniform(-0.05,0.05) for i in range(parameters[m])] for i in range(parameters[m+1])]) for m in range(len(parameters)-1)]
        
        self.b=[np.array([[0] for i in range(parameters[m+1])])for m in range(len(parameters)-1)]
        
        if use_para:
            self.w = [data[f"W{i}"] for i in range(len(self.w))]
            self.b = [data[f"b{i}"] for i in range(len(self.b))]
        
        
    def forward(self,in_layer,out_layer):
        return sigmoid(self.w[out_layer-1] @ in_layer+self.b[out_layer-1])
    
    def forwardz(self,in_layer,out_layer):
        return self.w[out_layer-1] @ in_layer+self.b[out_layer-1]
    
    def compute(self,input):
        z_l=[0]
        a_l=[]
        a_l.append(input.reshape(-1, 1))
        for _ in range(len(self.parameters)-1):
                z_l.append(self.forwardz(a_l[_],_+1))
                a_l.append(sigmoid(z_l[_+1]))
        return a_l[-1]
    
    def computesoftmax(self,input):
        z_l=[]
        a_l=[]
        a_l.append(input.reshape(-1,1))
        for layer_idx in range(len(self.parameters)-1):
                z_l.append(self.forwardz(a_l[layer_idx], layer_idx+1))

                if layer_idx == len(self.parameters)-2:  
                    if self.parameters[-1] == 1:
                        a_l.append(sigmoid(z_l[-1]))
                    else:
                        a_l.append(softmax(z_l[-1]))
                else:
                    a_l.append(sigmoid(z_l[-1]))
                    
        return a_l[-1]
    
    def compute_and_calculateMSE(self, input, wanted_output):
        output = self.compute(input)
        # If wanted_output is scalar, convert to one-hot
        if np.isscalar(wanted_output) or len(wanted_output) == 1:
            y_one_hot = np.zeros((self.parameters[-1], 1))
            y_one_hot[int(wanted_output)] = 1.0
            return np.mean((output - y_one_hot)**2)
        else:
            return np.mean((output - wanted_output.reshape(-1,1))**2)
        
    def compute_and_calculateMSE_soft(self, input, wanted_output):
        output = self.computesoftmax(input)
        # If wanted_output is scalar, convert to one-hot
        if np.isscalar(wanted_output) or len(wanted_output) == 1:
            if self.parameters[-1] == 1:
                y_target = np.array([[float(wanted_output)]])   # shape (1,1)
            else:
                y_target = np.zeros((self.parameters[-1], 1))
                y_target[int(wanted_output)] = 1.0
            
            return np.mean((output - y_target)**2)
        else:
            return np.mean((output - wanted_output.reshape(-1,1))**2)