import cv2 as cv
import numpy as np
# import pandas as pd
import matplotlib.pylab as plt
from PIL import Image
#from skimage.draw import circle
from skimage import io, color, data
# from skimage.feature import match_template 

urls = [
    "https://iiif.lib.ncsu.edu/iiif/0052574/full/800/0/default.jpg",
    "https://iiif.lib.ncsu.edu/iiif/0016007/full/800/0/default.jpg",
    "https://placekitten.com/800/571"
]

for n, url in enumerate(urls):
    plt.figure()
    image = io.imread(url)
    image_2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    final_frame = cv.hconcat((image, image_2))
    plt.imshow(final_frame)
    print('\n')
    plt.savefig(f'image_processing/img{ n} .png')

io.imshow(image)

print(image.shape)

print(image.dtype)

print(image.shape[0])

print(image.shape[1])

print(image.shape[2])

plt.savefig(f'image_processing/img3.png')

plt.hist(image.ravel(),bins = 256, range = [0,256])
plt.savefig(f'image_processing/img4.png')

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.savefig(f'image_processing/img5.png')

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

plt.imshow(gray_image)
plt.savefig(f'image_processing/img6.png')

plt.hist(gray_image.ravel(),bins = 256, range = [0, 256])
plt.savefig(f'image_processing/img7.png')

plt.contour(gray_image, origin = "image")
plt.savefig(f'image_processing/img8.png')

ret, thresh = cv.threshold(gray_image,150,255,0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
image = cv.drawContours(image, contours, -1, (0, 255, 0), 3)
result = Image.fromarray((image).astype(np.uint8))
result.save('image_processing/img9.png')

im2 = - gray_image + 255
result = Image.fromarray((im2).astype(np.uint8))
result.save('image_processing/img10.png')

im3 = gray_image + 50
result = Image.fromarray((im3).astype(np.uint8))
result.save('image_processing/img11.png')

def histeq(im, nbr_bins = 256):
    imhist, bins = np.histogram(im.flatten(), nbr_bins, [0, 256])
    cdf = imhist.cumsum() 
    cdf = imhist.max()*cdf/cdf.max()
    cdf_mask = np.ma.masked_equal(cdf, 0)
    cdf_mask = (cdf_mask - cdf_mask.min())*255/(cdf_mask.max() - cdf_mask.min())
    cdf = np.ma.filled(cdf_mask,0).astype('uint8')
    return cdf[im.astype('uint8')]

im5 = histeq(im3)
plt.imshow(im5)
plt.show()


