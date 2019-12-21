from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np



fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.linspace(-50, 50, 1000)
y = np.linspace(-50, 50, 1000)
X, Y = np.meshgrid(x, y)

def f(x, y):
    x = np.array([x, y])
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(np.power(x, 2)))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.exp(1)

def render_z(X, Y, f):
    d1, d2 = X.shape
    Z = np.empty([d1, d2])
    for i in range(d1):
        for j in range(d2):
            Z[i, j] = f(X[i, j], Y[i, j])
    return Z

Z = render_z(X, Y, f)

#X, Y, Z = axes3d.get_test_data(0.05)

ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-50, 50)
ax.set_ylabel('Y')
ax.set_ylim(-50, 50)
ax.set_zlabel('Z')
#ax.set_zlim(-100, 100)

plt.show()

