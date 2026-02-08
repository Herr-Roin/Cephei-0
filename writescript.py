import os
import numpy as np

def writedataloop(): #Manually write training data
    contin=""
    storein=input("Name yo output file ")
    while contin=="":
        contin=input("Press Enter to continue ")
        
        #a_0,teacher=userinput.take_input(False)
        
        a_0 = np.asarray(a_0)

        if a_0.ndim == 2:
            a_0 = a_0.reshape(-1)   # (784, 1) -> (784,)

        teacher = int(teacher)
        
        if os.path.exists("data/"+ storein + ".npz"):
            with np.load("data/"+ storein + ".npz") as data:
                X = data["X"]
                Y = data["Y"]

            X = np.vstack([X, a_0])
            Y = np.append(Y, teacher)

        else:
            X = a_0[None, :]        # shape (1, 784)
            Y = np.array([teacher]) # shape (1,)

        np.savez("data/"+ storein + ".npz", X=X, Y=Y)

        print(f"Saved sample #{len(Y)} (label={teacher})")
        
        
        
def writedata(inp,teach):

    a_0,teacher=inp,teach
    
    a_0 = np.asarray(a_0)

    if a_0.ndim == 2:
        a_0 = a_0.reshape(-1)   # (784, 1) -> (784,)

    teacher = int(teacher)
    
    if os.path.exists("data/validationdata.npz"):
        with np.load("data/validationdata.npz") as data:
            X = data["X"]
            Y = data["Y"]

        X = np.vstack([X, a_0])
        Y = np.append(Y, teacher)

    else:
        X = a_0[None, :]        # shape (1, 784)
        Y = np.array([teacher]) # shape (1,)

    np.savez("data/validationdata.npz", X=X, Y=Y)

    print(f"Saved sample #{len(Y)} (label={teacher})")
    
#writedataloop()