
#**FODS Assignment 3**

##Preprocessing


import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm_notebook
import matplotlib.colors
from mpl_toolkits import mplot3d
import scipy.special
import matplotlib.ticker as mt

############## FOR GIF and Animation ##############
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML

np.random.seed(5824)
toss_array = np.random.randint(0,2,160)
print(np.mean(toss_array))
mean_data = np.mean(toss_array)

a = 2
b =3

gamma_ab = scipy.special.gamma (a+b)
gamma_a = scipy.special.gamma (a) 
gamma_b = scipy.special.gamma (b)

mu = np.linspace(0,1,50)

prior = (gamma_ab / ((gamma_a)*(gamma_b))) * ((mu)**(a-1)) * ((1-mu)**(b-1))

"""##Part A"""

m=0
l=0
posterior_matrix  =  []
posterior = prior
posterior_matrix.extend(posterior)
count =1
plt.plot(mu, posterior, '-*')
for i in range(1,160):
  if(toss_array[i]==1):
    m+=1
  else:
    l+=1
  gamma_final =  (scipy.special.gamma(m+a+l+b))/((scipy.special.gamma(m+a))*(scipy.special.gamma(l+b)))
  posterior = gamma_final * (mu**(m+a-1))*((1-mu)**(l+b-1))

  posterior_matrix.extend(posterior)
  if count%20==0:
    plt.plot(mu, posterior, '-*') 
    plt.xlabel('u_parameter')
    plt.ylabel('prior') 
    plt.title('uML = 0.65')
  count+=1
plt.show()

t = np.array(posterior_matrix)
t = t.reshape(160,50)

# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()

ax.set_xlim(0,1)
ax.set_ylim((0,12))
ax.set_xlabel('u_parameter')
ax.set_ylabel('prior')

line, = ax.plot([], [], lw=2)


# animation function. This is called sequentially
def animate(i):
    x = mu
    y = t[i,:]
    line.set_data(x, y)
    return (line,)

    # call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate,frames=160, interval=50, blit=True)
#For converting to video
HTML(anim.to_html5_video())

"""##Part B"""

m=0
l=0
posterior = []

a=2
b=3

for i in range(1,160):
  if(toss_array[i]==1):
    m+=1
  else:
    l+=1

gamma_final =  (scipy.special.gamma(m+a+l+b))/((scipy.special.gamma(m+a))*(scipy.special.gamma(l+b)))
posterior = gamma_final * (mu**(m+a-1))*((1-mu)**(l+b-1))

plt.plot(mu, posterior, '-*') 
plt.xlabel('u_parameter')
plt.ylabel('prior') 
plt.title('uML = 0.65')
plt.show()

max=0
for i in range(len(mu)):
  if posterior[i]>max:
    max=posterior[i]
    value = mu[i]
print(max , value)