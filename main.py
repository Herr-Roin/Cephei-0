import numpy as np
import network
import visualize
import userinput

data=np.load("data/params.npz")
net=network.Network([28*28,100,10],True,data)
#net=network.Network([28*28,16,1],False,0)

def main(netw):
    net=netw
    iterate="Y"


    #visualize.draw_neural_net(1, .1, .9, .01, .9, [4, 8, 1])

    while iterate=="Y":
        iterate=input("Continue Y/N")
        a_0=userinput.take_input(False)
        output=net.computesoftmax(a_0)
        ind=list(output).index(np.max(output))
        
        print(output)
        if net.parameters[-1]!=1:
         print(f"Your drawing looks like a {ind}")

valdata = np.load("data/validationdata.npz")
X_test, Y_test = valdata["X"], valdata["Y"]

correct = 0
for i in range(len(X_test)):
    output = net.computesoftmax(X_test[i])
    pred = np.argmax(output)
    if pred == Y_test[i]:
        correct += 1

print(f"Test Accuracy: {correct/len(X_test):.2%}")
        
main(net)    
print("end")
