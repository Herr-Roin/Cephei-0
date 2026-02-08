import numpy as np

def sigmoid(x):
    return 1/(1+np.e**-x)

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
    
    def compute_and_calculateMSE(self,input,wanted_output):
        output=self.compute(input)
        squared_differences=0
        for i in range(len(output)):
            try: squared_differences+=(output[i]-wanted_output[i])**2
            except IndexError: squared_differences+=(output-wanted_output)**2
            
        return squared_differences/len(output)