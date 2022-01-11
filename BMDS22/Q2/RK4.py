import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def f(step, subtrate, complex):
    # derivative function of S
    k1 = 100/60
    k2 = 600/60
    E0 = 1
    lamda = k1 * E0
    # this the equation we want to solve
    fs = -lamda*subtrate + (k1*subtrate + k2)*complex
    return fs

def g(step, subtrate, complex):
    # derivative function of C
    k1 = 100/60
    k2 = 600/60
    k3 = 150/60
    E0 = 1
    theta = k2 + k3
    lamda = k1 * E0
    # this the equation we want to solve
    fc = lamda*subtrate - (theta + k1*subtrate)*complex
    return fc


# RK4
# define the initial values
E0 = 1
k3 = 150/60
# define step width and initial values of C and S
h = 0.001  # step width
n = int(30/0.001)  # number of steps, assume it took 30s
t = []  # put each step in a list
temp = 0
for i in range (1,n+1):
    temp = h*i
    t.append(temp)
# put the initial value of C and S in lists
c = [0]
s = [10]
# RK4 calculation
for k in range(0,n-1):
    z0 = h * f(t[k], s[k], c[k])
    l0 = h * g(t[k], s[k], c[k])
    z1 = h * f(t[k]+h/2, s[k]+z0/2, c[k]+l0/2)
    l1 = h * g(t[k]+h/2, s[k]+z0/2, c[k]+l0/2)
    z2 = h * f(t[k]+h/2, s[k]+z1/2, c[k]+l1/2)
    l2 = h * g(t[k]+h/2, s[k]+z1/2, c[k]+l1/2)
    z3 = h * f(t[k]+h, s[k]+z2, c[k]+l2)
    l3 = h * g(t[k]+h, s[k]+z2, c[k]+l2)
    s_temp = s[k] + (z0+2*z1+2*z2+z3)/6
    c_temp = c[k] + (l0+2*l1+2*l2+l3)/6
    # add new s/c to their own list
    s.append(s_temp)
    c.append(c_temp)


# solution to E,P,Vp
# solution to E
e = []
for item in c:
    e_value = E0 - item
    e.append(e_value)
# solution to P
p = []
sum = 0
for k in range(0,n) :
    sum = c[k] + sum
    p_value = k3 * sum * h  # integrating discrete results
    p.append(p_value)
# solution to the change rate of P
Vp = []
for item in c:
    Vp_value = k3*item
    Vp.append(Vp_value)


# output results
result = np.array([e,s,c,p])
data = result.T
data_df = pd.DataFrame(data)
data_df.columns = ['enzyme', 'substract', 'compound', 'product']
data_df.to_csv('concentration per step(0.001s).csv')


# plot
# figure 1: concentration - time
plt.axis([0, 35, 0, 15])  # range of x,y
plt.xlabel('reaction time(s)')
plt.ylabel('concentration(uM)')
line1, = plt.plot(t, e, color='r', linewidth=1)
line2, = plt.plot(t, s, color='m', linewidth=1)
line3, = plt.plot(t, c, color='y', linewidth=1)
line4, = plt.plot(t, p, color='g', linewidth=1)
plt.legend(loc='best', fontsize=6, handles=[line1, line2, line3, line4],labels=['enzyme', 'substrate', 'compound', 'product'])
plt.show()
# figure 2: concentration of S - change rate of P
plt.axis([0, 12, 0, 2])  # range of x,y
plt.xlabel('concentration of substrate(uM)')
plt.ylabel('change rate of P(uM/s)')
line5, = plt.plot(s, Vp, color='r', linewidth=1)
plt.show()
