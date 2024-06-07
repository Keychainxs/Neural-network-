import numpy as np
import matplotlib.pyplot as plt    


x = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
y = np.array([[0,1,1,0]]).T

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) -1 

for i in range(70000): 
    # foward prop
    l1 = 1/ (1+ np.exp(-(np.dot(x,syn0))))
    l2 =  1/ (1+ np.exp(-(np.dot(l1,syn1))))

    #backwards prop
    l2_outPut = (y-l2) * (l2 *(1 -l2))

    l1_outPut = l2_outPut.dot(syn1.T) * (l1*(1-l1))

   

    syn1 += l1.T.dot(l2_outPut)
    syn0 += x.T.dot(l1_outPut)
    # create the grid for the bodaries 
    xx, yy = np.meshgrid(np.linespace(0,1,100),np.linespace (0,1,100))
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx.ravel)]

    reshape=l2.reshape(xx.shape)

    plt.contour(xx, yy, reshape, levels = [0, 0.5, 1], alpha = 0.2, colors = ['red','blue'])
    plt.scatter(x[:, 0], x[:, 1], c= y[:, 0], s = 50, cmap ='bwr', edgecolor = 'k')
    plt.title('XOR Neural Network Decision Boundary')
    plt.show()
print("predicted output for the training model: ")
print(l2)
print("the final wieghts after trainig: ")

print("syn0: ", syn0)
print("syn1: ", syn1)





