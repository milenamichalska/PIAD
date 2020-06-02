import numpy as np

#tablice
a = np.array([1,2,3,4,5,6,7])
b = np.array([[1,2,3,4,5], [6,7,8,9,10]])
#print(b)
np.transpose(b)
#print(b)

c = np.arange(100)
#print(c)

d = np.linspace(0, 2.0, num=10)
#print(d)

e = np.arange(0, 100, 5)
#print(e)

#liczby losowe
f = np.around(np.random.random_sample((20,)), decimals=2)
#print(f)

g = np.random.choice(np.arange(1000), 100)
#print(g)

h = np.zeros((3, 2))
#print(h)

i = np.ones((3, 2))
#print(i)

j = np.random.randint(100, size=(5, 5), dtype=np.int32)
#print(j)

#zadanie
a1 = np.random.choice(np.linspace(0, 10.0), 100)
b1 = a1.astype(int)
a1 = np.around(a1)
a1 = a1.astype(int)

#print(a1)
#print(b1)

#selekcja danych
b2 = np.array([[1,2,3,4,5], [6,7,8,9,10]],dtype=np.int32)
#print(b2.ndim)
#print(b2.size)
#print(b2)
#b2a = np.select([2,4], b2)
#print(b2a)
#print(b2[:1,])
#print(b2[:,:1])
b2b = np.around(np.random.uniform(low=0, high=100, size=(20,7)))
#print(b2b[:4])

#operacje matematyczne i logiczne
k = np.random.uniform(low=0, high=10, size=(3,3))
l = np.random.uniform(low=0, high=10, size=(3,3))
# print (k + l)
# print (k * l)
# print (k.dot(l))
# print (k / l)
# print (np.sum(k) >= 4)
# print (l.diagonal())

#dane statystyczne
# print (np.sum(l))
# print (np.min(l))
# print (np.max(l))
# print (np.mean(l))
# print (l.mean(0))
# print (l.mean(1))

#rzutowanie danych
m = np.arange(50)
# print (m.reshape((10, 5)))
# print (np.resize(m, (10,5)))
# print (np.ravel(m))
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([5, 4, 3, 2])
x1n = x1[:, np.newaxis]
x2n = x2[:, np.newaxis]

#sortowanie danych
p = np.random.randn(5,5)
# print (np.sort(p, axis=0))
# print (np.argsort(p, axis=0)[::-1])

#zadania podsumowujące
#zadanie1
x = np.random.randint(1000, size=(10, 5))
#print(x)
print (np.trace(x))

#zadanie2
x1 = np.random.random_sample((5,))
x2 = np.random.random_sample((5,))
print (x1*x2)

#zadanie3
x3 = np.random.choice(np.arange(100), 25).reshape((5, 5))
x4 = np.random.choice(np.arange(100), 25).reshape((5, 5))
print (x3+x4)

#zadanie4
x3 = np.random.choice(np.arange(100), 20).reshape((5, 4))
x4 = np.random.choice(np.arange(100), 20).reshape((4, 5))
x4n = x4.reshape((5,4))
print (x3 + x4n)

#zadanie5
print (x3[3:4,:] * x4n[3:4,:])

#zadanie6
mu, sigma = 0, 0.1 # mean and standard deviation
s1 = np.random.normal(mu, sigma, 1000)
s2 = np.random.uniform(-1,0,1000)

print (np.mean(s1))
print (np.mean(s2))
print (np.std(s1))
print (np.std(s2))
print (np.var(s1))
print (np.var(s2))

#zadanie7
a = np.random.randint(100, size=(5, 5))
b = np.random.randint(100, size=(5, 5))
print (a*b)
print (a.dot(b))

#zadanie8
print (a)
print (a.strides)

#zadanie9
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
print (np.vstack((a,b)))
print (np.hstack((a,b)))
