import numpy as np
import matplotlib.pylab as plt

x = np.linspace(1,100,100)
y = np.sin(x) + np.random.random(100)
plt.plot(x, y)
plt.show()


