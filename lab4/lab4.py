import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

#dyskretyzacja
def discretizeSinus(f, Fs):
    n = 10 #liczba cykli
    t = n / f #wzór na częstotliwość
    Ts = 1. / Fs
    N = int(t / Ts) #liczba próbek
    print(N)
    
    x = np.linspace(0, t, N)
    y = np.sin(2*np.pi*f*x)
    plt.plot(x, y)
    plt.show()

# discretizeSinus(10, 20)
# discretizeSinus(10, 21)
# discretizeSinus(10, 30)
# discretizeSinus(10, 45)
# discretizeSinus(10, 50)
# discretizeSinus(10, 100)
# discretizeSinus(10, 200)
# discretizeSinus(10, 250)
# discretizeSinus(10, 1000)

#Twierdzenie o próbkowaniu, twierdzenie Nyquista–Shannona
#Częstotliwość sygnału nie powinna przekraczać połowy częstotliwości próbkowania.

#Aliasing
#Nieodwracalne zniekształcenie sygnału w procesie próbkowania wynikające z niespełnienia założeń twierdzenia o próbkowaniu.

methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

# Fixing random state for reproducibility
np.random.seed(19680801)

img = mpimg.imread (r'C:/Users/eversis2/Documents/studia/PIAD/lab4/pluto.png')
plt.imshow(img)
plt.show()

# fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
#                         subplot_kw={'xticks': [], 'yticks': []})

# for ax, interp_method in zip(axs.flat, methods):
#     ax.imshow(img, interpolation=interp_method, cmap='viridis')
#     ax.set_title(str(interp_method))

# plt.tight_layout()
# plt.savefig('interpolation_methods.png')

#kwantyzacja
a = plt.imshow(img)
arr = a.get_array()
print(arr.shape)
#(max(rgb[... , :3]) + min(rgb[... , :3])/2)

jasnosc = lambda rgb : np.dot(rgb[...,:3],[(rgb.max()+rgb.min()/2),(rgb.max()+rgb.min()/2),(rgb.max()+rgb.min()/2)])
a1 = jasnosc(img)  
plt.imshow(a1)
plt.show() 

usrednienie = lambda rgb : np.dot(rgb[...,:3],[(rgb.sum()/3),(rgb.sum()/3),(rgb.sum()/3)])
a2 = usrednienie(img)  
plt.imshow(a2)
plt.show() 

luminacja = lambda rgb : np.dot(rgb[... , :3] , [0.21 , 0.72, 0.07]) 
a3 = luminacja(img)  
plt.imshow(a3)
plt.show()

histogram1, bin_edges1 = np.histogram(a1, bins=256, range=(0, 1))
plt.figure()
plt.plot(bin_edges1[0:-1], histogram1)
plt.show()

histogram2, bin_edges2 = np.histogram(a2, bins=256, range=(0, 1))
plt.figure()
plt.plot(bin_edges2[0:-1], histogram2)
plt.show()

histogram3, bin_edges3 = np.histogram(a3, bins=256, range=(0, 1))
plt.figure()
plt.plot(bin_edges3[0:-1], histogram3)
plt.show()

histogram4, bin_edges4 = np.histogram(a3, bins=16, range=(0, 1))
plt.figure()
plt.plot(bin_edges4[0:-1], histogram4)
plt.show()

print(histogram4)
