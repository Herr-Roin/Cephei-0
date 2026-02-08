import numpy as np
import network
import visualize
import userinput

data=np.load("data/params.npz")
net=network.Network([28*28,32,1],True,data)
#net=network.Network([28*28,16,1],False,0)

def main(netw):
    net=netw
    iterate="Y"


    visualize.draw_neural_net(1, .1, .9, .01, .9, [4, 8, 1])

    while iterate=="Y":
        iterate=input("Continue Y/N")
        a_0=userinput.take_input(False)
        a_1=net.forward(a_0,1)
        output=net.forward(a_1,2)
        print(output)
        
main(net)    
print("end")
