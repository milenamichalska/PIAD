import pandas as pd
import numpy as np

# manipulowanie danymi
df1 = pd.DataFrame(np.random.random_sample((5,3)), columns=['A', 'B', 'C'], index=pd.date_range(start="2020-03-01",end="2020-03-05"))
df1.index.name = 'data'
#print(df1)

a = pd.DataFrame(np.random.random_sample((20,3)), columns=['A', 'B', 'C'], index=np.arange(20))
a.index.name = 'id'
# print(a)
#wybieranie z tabeli
# print(a.iloc[:3])
# print(a.iloc[-3:])
# print(a.index.name)
# print(a.columns.values)
# print(a.values)
# print(a.sample(n=5))
# print(a[['A']])
# print(a[['A', 'B']])
# print(a[['A', 'B']].iloc[:3])
# print(a.iloc[[5]])
# print(a.iloc[[0, 5, 6, 7], :2])

#describe
# print(a[a>0].describe())
# print(a[a>0])
# print(a[['A']].describe())
# print(a.mean())
# print(a.apply(pd.DataFrame.mean, axis=1))

#concat
b = pd.DataFrame(np.random.random_sample((20,3)), columns=['A', 'B', 'C'], index=np.arange(20))
c = pd.DataFrame(np.random.random_sample((20,3)), columns=['A', 'B', 'C'], index=np.arange(20))
# print(pd.concat([b,c]))

#sortowanie
df = pd.DataFrame(data={'x': [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']}, index=np.arange(5))
df.index.name = 'id'
# print(df)
# print(df.sort_index)
# print(df.sort_values(by=['y'], ascending=0))

#grupowanie danych
slownik = {'Day': ['Mon', 'Tue', 'Mon', 'Tue', 'Mon'], 'Fruit': ['Apple', 'Apple', 'Banana', 'Banana', 'Apple'], 'Pound': [10, 15, 50, 40, 5], 'Profit':[20, 30, 25, 20, 10]}
df3 = pd.DataFrame(slownik)
# print(df3)
# print(df3.groupby('Day').sum())
# print(df3.groupby(['Day','Fruit']).sum())

# wypełnianie danych
df2 = pd.DataFrame(np.random.randn(20, 3), index=np.arange(20), columns=['A','B','C'])
df2.index.name = 'id'
#print(df2)
#Wykonaj i opisz jak działają poniższe komendy:
df2['B'] = 1
#print(df2)
#zmienia wszystkie wartości dla kolumny B na 1
df2.iloc[1,2] = 10
#print(df2)
# df2[df2:0] = -df2
# print(df2)

#uzupełnienie danych
#TODO

df = pd.DataFrame(data={'x': [1, 2, 3, 4, 5], 'y': ['a', 'b', 'a', 'b', 'b']}, index=np.arange(5))
#zadanie 1
print(df.groupby('y').mean())
#zadanie 2
print(df['x'].value_counts())
print(df['y'].value_counts())
#zadanie 3
#data1 = np.loadtxt('autos.csv', delimiter=',')
#print(data1)
data2 = pd.read_csv('autos.csv')
print(data2)
#zadanie 4
print(data2.groupby('make')['city-mpg', 'highway-mpg'].mean().reset_index())
#zadanie 5
print(data2.groupby('make')['fuel-type'].value_counts())
#zadanie 6
x = data2['city-mpg']
y = data2['length']
print(np.polyfit(x, y, 1))
print(np.polyfit(x, y, 2))
#zadanie 7
from scipy import stats
print(stats.pearsonr(data2['city-mpg'], data2['length']))
#zadanie 8
from matplotlib import pyplot as plt
trend = np.polyfit(x, y, 1)
plt.plot(x,y,'o')
trendpoly = np.poly1d(trend) 
plt.plot(x,trendpoly(x))
plt.show()
#zadanie 9
xs = np.linspace(0, 50, data2['length'].count())
plt.plot(xs, y,'o')
xs = np.linspace(0, 50, 200)
density = stats.gaussian_kde(data2['length'])
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.plot(xs, density(xs), label="gęstość")
plt.legend(loc='upper right')
plt.show()
#zadanie 10
xs = np.linspace(0, 50, data2['length'].count())
plt.plot(xs, y)
xs = np.linspace(0, 50, data2['width'].count())
plt.plot(xs, data2['width'])
plt.show()
#zadanie 11
xd = data2['length']
yd = data2['width']

xy = np.vstack([xd, yd])
kernel = stats.gaussian_kde(xy, bw_method='silverman')
xmin = xd.min()
xmax = xd.max()
ymin = yd.min()
ymax = yd.max()
    
X, Y = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kernel(positions).T, X.shape)
plt.contour(X, Y, Z)
plt.savefig('gestosc.png')
plt.savefig('gestosc.pdf')
