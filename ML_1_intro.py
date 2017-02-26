import numpy as np

'''
a = np.array([0, 1, 2, 3, 4, 5])
b = a.reshape((3,2))
b[1][0] = 77
c = a.reshape((3,2)).copy()
c[0][0] = -99

d = np.arange(6)

#print(d*2)
#print(d**2)

#print(a)
#print(a[np.array([2, 3, 4])])

print(a)
print(a[a>4])
a[a>4] = 4
print(a.clip(0,4))
'''

'''
c = np.array([1, 2, np.NAN, 3, 4])
print(np.isnan(c))
print(c[~np.isnan(c)])
print(np.mean(c[~np.isnan(c)]))
'''

'''
import timeit

normal_py_sec = timeit.timeit('sum(x*x for x in range(1000))', number=10000)

naive_np_sec = timeit.timeit(
    'sum(na*na)', setup="import numpy as np; na=np.arange(1000)", number=10000)

good_np_sec = timeit.timeit(
    'na.dot(na)', setup="import numpy as np; na=np.arange(1000)", number=10000)

print("Normal Python: %f sec" % normal_py_sec)
print("Naive NumPy: %f sec" % naive_np_sec)
print("Good NumPy: %f sec" % good_np_sec)
'''


'''
a = np.array([1., 2., 3.])
print(a.dtype)
a = np.array([1, "stringly"])
print(a.dtype)
'''

import scipy as sp
import matplotlib.pyplot as plt


def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)


def plot_models(x, y, models, mx=None):

    plt.scatter(x, y, s=5)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks([w * 7 * 24 for w in range(10)],
               ['week %i' % w for w in range(10)])

    if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000)
        for model in models:
            plt.plot(mx, model(mx), linewidth=2)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    plt.ylim(ymax=10000)
    plt.xlim(xmin=0)
    plt.grid(True, linestyle='-', color='0.75')
    plt.show()




data = sp.genfromtxt("web_traffic.txt", delimiter="\t")
#print(data.shape)
x = data[:, 0]
y = data[:, 1]
print(sp.sum(sp.isnan(y)))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]


#plot_models(x, y, None)


fp1, res1, rank1, sv1, rcond1 = sp.polyfit(x, y, 1, full=True)
f1 = sp.poly1d(fp1)

fp2, res2, rank2, sv2, rcond2 = sp.polyfit(x, y, 2, full=True)
f2 = sp.poly1d(fp2)

f3 = sp.poly1d(sp.polyfit(x, y, 3))
f10 = sp.poly1d(sp.polyfit(x, y, 10))

print("Errors for the complete data set: ")
for f in [f1, f2, f3, f10]:
    print("Error d=%i: %f" % (f.order, error(f, x, y)))

inflection = 3.5 * 7 * 24
inflection = int(inflection)
xa = x[:inflection]
ya = y[:inflection]

xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, xa, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)

print("Errors after the inflection point: ")
for f in [f1, f2, f3, f10]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))

print("Error inflection = %f" % (fa_error + fb_error))

fb1 = fb
fb2 = sp.poly1d(sp.polyfit(xb, yb, 2))
fb3 = sp.poly1d(sp.polyfit(xb, yb, 3))
fb10 = sp.poly1d(sp.polyfit(xb, yb, 10))

print("Errors for only the time after inflection point")
for f in [fb1, fb2, fb3, fb10]:
    print("Error d=%i: %f" % (f.order, error(f, xb, yb)))


from  scipy.optimize import fsolve
reached_max = fsolve(fb2 - 100000, x0=800)/(7*24)
print("100,000 hits/hour expected at week %f" % reached_max[0])

plot_models(x, y, [fb1, fb2, fb3, fb10, fa, fb],
            mx=sp.linspace(0 * 7 * 24, 6 * 7 * 24, 100))



