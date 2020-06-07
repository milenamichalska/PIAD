import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
import scipy

#klasteryzacja

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# print(X)

from sklearn.cluster import AgglomerativeClustering
Y_ward = AgglomerativeClustering(linkage='ward', n_clusters=3).fit(X).labels_ #metoda Warda - minimalizuje wariancję między dwoma klastrami
# print(Y_ward)
Y_avg = AgglomerativeClustering(linkage='average', n_clusters=3).fit(X).labels_ #metoda średnich połączeń
print(Y_avg) 
Y_single = AgglomerativeClustering(linkage='single', n_clusters=3).fit(X).labels_  #metoda najbliższego sasiedztwa
# print(Y_single)
Y_complete = AgglomerativeClustering(linkage='complete', n_clusters=3).fit(X).labels_ #metoda najdalszych połączeń
# print(Y_complete) 

# print(Y)

def ﬁnd_perm(clusters, Y_real, Y_pred):
    perm = []
    for i in range(clusters):
        idx = Y_pred == i
        new_label = scipy.stats.mode(Y_real[idx])[0][0]
        perm.append(new_label)
    return [perm[label] for label in Y_pred]

# print(find_perm(3, Y, Y_avg))
# print(find_perm(3, Y, Y_ward))
# print(find_perm(3, Y, Y_single))
# print(find_perm(3, Y, Y_complete))

from sklearn.metrics import jaccard_score
#Współczynnik Jaccarda mierzy podobieństwo między dwoma zbiorami i jest zdefiniowany jako iloraz mocy części wspólnej zbiorów i mocy sumy tych zbiorów:

print(jaccard_score(Y, Y_avg, average=None))
print(jaccard_score(Y, Y_ward, average=None))
print(jaccard_score(Y, Y_single, average=None))
print(jaccard_score(Y, Y_complete, average=None))

from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, convex_hull_plot_2d

pca = PCA(n_components=2)
X_reduced = pca.ﬁt_transform(X)
# print(X_reduced)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.set_title("oryginal")
colors = ['navy', 'turquoise', 'darkorange']

for color, i in zip(colors, [0, 1, 2]):
        hull = ConvexHull(X_reduced[Y == i])
        ax1.scatter(X_reduced[Y == i, 0], X_reduced[Y == i, 1], color=color, lw=2)
        for simplex in hull.simplices:
            ax1.plot(X_reduced[Y == i][simplex, 0], X_reduced[Y == i][simplex, 1], 'k-', color=color)

ax2.set_title("rezultat klasteryzacji")

for color, i in zip(colors, [0, 1, 2]):
        hull = ConvexHull(X_reduced[Y_avg == i])
        ax2.scatter(X_reduced[Y_avg == i, 0], X_reduced[Y_avg == i, 1], color=color, lw=2)
        for simplex in hull.simplices:
            ax2.plot(X_reduced[Y_avg == i][simplex, 0], X_reduced[Y_avg == i][simplex, 1], 'k-', color=color)

ax3.set_title("roznica")
for i, points in enumerate(X_reduced):
    color = 'green'
    if (Y[i] != Y_avg[i]):
        color = 'red'
    ax3.scatter(points[0], points[1], color=color)

plt.show()

from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)
X_reduced = pca.ﬁt_transform(X)

fig1, (ax4, ax5, ax6) = plt.subplots(1, 3)

ax4 = fig1.add_subplot(1, 3, 1, projection='3d')
ax4.set_title("oryginal")

for color, i in zip(colors, [0, 1, 2]):
    ax4.scatter(X_reduced[Y == i, 0], X_reduced[Y == i,  1], X_reduced[Y == i, 2], color=color)

ax5 = fig1.add_subplot(1, 3, 2, projection='3d')
ax5.set_title("rezultat klasteryzacji")

for color, i in zip(colors, [0, 1, 2]):
    ax5.scatter(X_reduced[Y_avg == i, 0], X_reduced[Y_avg == i,  1], X_reduced[Y_avg == i, 2], color=color)

ax6 = fig1.add_subplot(1, 3, 3, projection='3d')
ax6.set_title("roznica")

for i, points in enumerate(X_reduced):
    color = 'green'
    if (Y[i] != Y_avg[i]):
        color = 'red'
    ax6.scatter(points[0], points[1], points[2], color=color)

plt.show()

#kwantyzacja
import matplotlib.image as mpimg
img = mpimg.imread (r'C:/Users/eversis2/Documents/studia/PIAD/lab7/slonecznik.jpg')
plt.imshow(img)
plt.show()

a = plt.imshow(img)
arr = a.get_array()
print(arr.shape)

new_arr = arr.reshape((arr.shape[0]*arr.shape[1]), arr.shape[2])
print(new_arr.shape)

new_arr_helper = new_arr

from sklearn.cluster import KMeans
kmeans3 = KMeans(n_clusters=3, random_state=0).fit(new_arr)
print(kmeans3.labels_)
print(kmeans3.labels_.shape)
print(kmeans3.cluster_centers_)

for i, pixels in enumerate(new_arr):
    new_arr[i] = kmeans3.cluster_centers_[kmeans3.labels_[i]]

new_arr = np.reshape(new_arr, (480, 640, 3))
print(new_arr.shape)
plt.imshow(new_arr)
plt.show()

from sklearn.metrics import mean_squared_error

err_arr = np.ones((480,640))

for i, pixels in enumerate(new_arr):
    err_arr[i] = np.square(np.subtract(new_arr_helper[i],new_arr[i])).mean() 

plt.imshow(err_arr)
plt.show()

#kmeans 5

a1 = plt.imshow(img)
arr1 = a1.get_array()
new_arr1 = arr1.reshape((arr1.shape[0]*arr1.shape[1]), arr1.shape[2])
new_arr_helper1 = new_arr

kmeans5 = KMeans(n_clusters=5, random_state=0).fit(new_arr1)

for i, pixels in enumerate(new_arr1):
    new_arr1[i] = kmeans5.cluster_centers_[kmeans5.labels_[i]]

new_arr1 = np.reshape(new_arr1, (480, 640, 3))
print(new_arr1.shape)
plt.imshow(new_arr1)
plt.show()

#kmeans 8

a2 = plt.imshow(img)
arr2 = a2.get_array()
new_arr2 = arr2.reshape((arr2.shape[0]*arr2.shape[1]), arr2.shape[2])

kmeans8 = KMeans(n_clusters=8, random_state=0).fit(new_arr2)

for i, pixels in enumerate(new_arr2):
    new_arr2[i] = kmeans8.cluster_centers_[kmeans8.labels_[i]]

new_arr2 = np.reshape(new_arr2, (480, 640, 3))
print(new_arr2.shape)
plt.imshow(new_arr2)
plt.show()
