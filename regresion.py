import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

data0 = pd.read_csv("pdata0.csv",nrows=100);
X = np.array(data0["x1"]);
Y = np.array(data0["x2"]);

def gradient_descent(x, y, alpha = 0.0001, iterations = 1000,  threshold = 1e-6):
    
    q0 = 0
    q1 = 0
    m = len(x)
    p_cost = None
    
    for i in range(iterations):
        hx = q0 + q1*x
        
        c_cost = np.sum((hx-y)**2) / len(y)
        if p_cost and abs(p_cost-c_cost)<=threshold:
            break  
        p_cost = c_cost
         
        q0 = q0 - alpha*(-(2/m)*sum(y-hx))
        q1 = q1 - alpha*(-(2/m)*sum(x*(y-hx)))
        
    return q0, q1
    
    q0, q1 = gradient_descent(X, Y, iterations=100);
    
hx = q1*X + q0;

plt.scatter(X, Y, marker='x', color='red')
plt.plot([min(X), max(X)], [min(hx), max(hx)], color='blue',markerfacecolor='red', markersize=10, linestyle='dashed')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
