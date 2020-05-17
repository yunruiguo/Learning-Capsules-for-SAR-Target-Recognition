import numpy
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
import re
import scipy.stats as stats

H = 100
ht = 92
hs = int(int(H-ht)/2)
def random_crop(img):

    image_crop = numpy.zeros(shape=(img.shape[0],ht,ht,3))
    for i in range(img.shape[0]):
        random_arr = numpy.random.randint(0, H - ht, size=2)
        y = int(random_arr[0])
        x = int(random_arr[1])
        image_crop[i,:,:,:] = img[i, y:y + ht, x:x + ht, :]
    return image_crop


def crop(img):
    image_crop = numpy.zeros(shape=(img.shape[0],ht,ht,3))
    for i in range(img.shape[0]):
        image_crop[i,:,:,:] = img[i, hs:hs+ht, hs:hs+ht, :]
    return image_crop


def additive_noise(img, var = 0.1):
    return random_noise(img, mode='gaussian', var=var)


def multiplicative_noise(img, sigma = 2, is_colour = 1):
    epa = stats.gamma.rvs(sigma, 0, sigma, (img.shape[0], img.shape[1], img.shape[2], img.shape[3]))
    # eps = numpy.random.rand(img.shape[0], img.shape[1], img.shape[2], img.shape[3])*sigma
    img = rescale_intensity(numpy.multiply(img, epa))
    # batch_plot(img)
    return img

def occlusion(img, var=0.2):
    num = img.shape[0]
    width = img.shape[1]
    img_size = var*width*width
    ratio = 0.5 + 0.5*numpy.random.rand()
    ocs1 = numpy.sqrt(img_size * ratio)
    ocs2 = numpy.sqrt(img_size / ratio)
    if numpy.random.binomial(1, 0.5):
        x = int(ocs1)
        y = int(ocs2)
    else:
        x = int(ocs2)
        y = int(ocs1)
    x = numpy.clip(x, 1, width)
    y = numpy.clip(y, 1, width)
    image_oclusion = numpy.copy(img)
    for i in range(num):
        start = numpy.random.randint(0, width, size=2)
        image_oclusion[i, start[0]:start[0]+x, start[1]:start[1]+y] = 0
    return image_oclusion

def enhance(img):
    """
        Custom preprocessing_function
    """
    img = img * 255
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.5))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.5))

    return np.array(img) / 255
def batch_plot(img):
    num = img.shape[0]
    for i in range(num):
        plt.imshow(img[i,:,:,:])
        plt.show()
    return