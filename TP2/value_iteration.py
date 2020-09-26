import numpy as np

def value_iteration(x, y, gamma):
    T_a = np.zeros((3,4,4))
    T_a[0] = [[0, 0, 0, 0], [0, 1-x, 0, x], [1-y, 0, 0, y], [1, 0, 0, 0]]
    T_a[1] = [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    T_a[2] = [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  
    U = np.array([0, 0, 0, 0])
    R = np.array([0, 0, 1, 10])

    while(1):
        if_break = np.zeros(4)
        U_next = np.zeros(4)
        T_sum = np.zeros((4,3))
        for i in range(4):# for S
            for j in range(3):# for T_a
                T_sum[i][j] = np.sum(T_a[j][i]*U)
            U_next[i] = R[i] + gamma*np.amax(T_sum[i])
            if U_next[i] - U[i] < 0.0001:
                if_break[i] = 1
        if (if_break.all()):
            break
        else:
            U = U_next
    print("The V* for all states are: ")      
    print("V*(S0)=%f, V*(S1)=%f, V*(S2)=%f, V*(S3)=%f" %(U[0],U[1],U[2], U[3]))
    print("The optimal policies are:")
    print("π*(S0)=a%d, π*(S1)=a%d, π*(S2)=a%d, π*(S3)=a%d" %(T_sum[0].argmax(), T_sum[1].argmax(), T_sum[2].argmax(),T_sum[3].argmax()))          

if __name__=='__main__':
    value_iteration(0.25, 0.25, 0.9)
