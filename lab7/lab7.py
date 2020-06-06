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
print(X_reduced)

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
