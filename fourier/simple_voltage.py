# 1.4.d

from math import cos, pi

import numpy as np
import matplotlib.pyplot as plt


def vt(v1, v2):
    twopi = 2 * pi
    return lambda t: 3 * cos(twopi * v1 * t - 1.3) + \
                     5 * cos(twopi * v2 * t + 0.5)

v1, v2 = 2, 1
v = vt(v1, v2)

n = 10
x = np.arange(0, n, 0.01)
y = map(v, x)

plt.plot(x, y)
plt.savefig('figs/1.4.d.png')
plt.show()

print('Max: %f, Min: %f' % (max(y), min(y)))