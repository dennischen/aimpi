from utils import basename_noext, kmp_duplicate_lib_ok

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('sample.png')
print(f'>>A {img.shape}, {img}')
fl = img.flatten()
print(f'>>AA {fl.shape}, {fl}')
plt.imshow(img)
plt.savefig(f'out/{basename_noext(__file__)}_1.png')

img = mpimg.imread('sample.jpg')
# print(f'>>B{img.shape}, {img}')
plt.savefig(f'out/{basename_noext(__file__)}_2.png')
