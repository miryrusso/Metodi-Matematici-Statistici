import numpy as np
DATA = np.array([[0., 19.5],[1., 22.1],[2., 24.3],[3., 25.7],[4., 26.1],[5., 28.5],
                [6., 30.],[7., 32.1],[8., 32.7],[9., 32.7],[10., 35.]])

x = DATA[:,0]    # velocità
y = DATA[:,1]    # ossigeno
n = x.size
display(x,y)

# Calcolo dei coefficienti di regressione y = b_0 + b_1 x
x_bar = np.mean(x)
y_bar = np.mean(y) 

sig_xy = np.sum((x-x_bar)*(y-y_bar))/n
display(sig_xy)

sig_x_2 = np.sum((x-x_bar)**2.)/n
display(sig_x_2)

b_0 = y_bar - sig_xy/sig_x_2*x_bar
b_1 = sig_xy/sig_x_2
display(b_0, b_1)

xx = np.linspace(0.,10.,100)
yy = b_0 + b_1*xx

import matplotlib.pyplot as plt
plt.plot(x, y, '*')
plt.plot(xx, yy)
plt.xlabel('velocità')
plt.ylabel('ossigeno')
plt.show()

# Calcolo dei residui
y_hat = b_0 + b_1*x
r = y - y_hat

plt.plot(x, r, '*')
plt.xlabel('velocità')
plt.ylabel('residuo')
plt.show()

from scipy.stats import probplot
from scipy.stats import norm
fig, ax = plt.subplots(1, 1)
probplot(r, dist=norm, plot=ax)
plt.show()

s2 = np.sum(r**2.)/(n-2)
display(s2)

# Test di significatività
# H0 : beta_1 = 0
# H1 : beta_1 <> 0

from scipy.stats import t
alpha = 0.05
T1 = np.sqrt(n)*b_1*np.sqrt(sig_x_2)/np.sqrt(s2)
display(T1)
T = t.ppf(1.-alpha/2.,n-2)
display(T)

# Calcolo del coefficiente di determinazione
sig_y_2 = np.sum((y-y_bar)**2.)/n
R2 = sig_xy**2./(sig_x_2*sig_y_2)
display(R2)