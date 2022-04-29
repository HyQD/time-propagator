import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-10,10,1000)

s = np.sin(t)
c = np.cos(t-np.pi/2)

plt.plot(t,s)
plt.plot(t,c,'--')
plt.show()