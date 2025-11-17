import math
import numpy as np
import scipy.ndimage.filters as ft
import torch
import torch.nn as nn
from torchvision.transforms.functional import pad
from torch.nn.functional import conv2d

def interp23tap(img, ratio):
    """
        Polynomial (with 23 coefficients) interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.


        Return
        ------
        img : Numpy array
            the interpolated img.

        """

    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'
    r, c, b = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)

    for z in range(int(ratio / 2)):

        I1LRU = np.zeros(((2 ** (z + 1)) * r, (2 ** (z + 1)) * c, b))

        if z == 0:
            I1LRU[1::2, 1::2, :] = img
        else:
            I1LRU[::2, ::2, :] = img

        for i in range(b):
            temp = ft.convolve(np.transpose(I1LRU[:, :, i]), BaseCoeff, mode='wrap')
            I1LRU[:, :, i] = ft.convolve(np.transpose(temp), BaseCoeff, mode='wrap')

        img = I1LRU

    return img


def interp23tap_torch(img, ratio):
    """
        A PyTorch implementation of the Polynomial interpolator Function.

        For more details please refers to:

        [1]  B. Aiazzi, L. Alparone, S. Baronti, and A. Garzelli - Context-driven fusion of high spatial and spectral
             resolution images based on oversampled multiresolution analysis
        [2] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone - Bi-cubic interpolation for shift-free pan-sharpening
        [3] G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. Orn Ulfarsson, L. Alparone, J. Chanussot -
            A new benchmark based on recent advances in multispectral pansharpening: Revisiting pansharpening with
            classical and emerging pansharpening methods


        Parameters
        ----------
        img : Numpy Array
            Image to be scaled. The conversion in Torch Tensor is made within the function. Dimension: H, W, B
        ratio : int
            The desired scale. It must be a factor power of 2.
        device : Torch Device
            The device on which perform the operation.


        Return
        ------
        img : Numpy array
           The interpolated img.

    """
    device = img.device
    assert ((2 ** (round(math.log(ratio, 2)))) == ratio), 'Error: Only resize factors power of 2'

    bs, b, r, c = img.shape

    CDF23 = np.asarray(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = [element * 2 for element in CDF23]
    BaseCoeff = np.expand_dims(np.concatenate([np.flip(CDF23[1:]), CDF23]), axis=-1)
    BaseCoeff = np.expand_dims(BaseCoeff, axis=(0, 1))
    BaseCoeff = np.concatenate([BaseCoeff] * b, axis=0)

    BaseCoeff = torch.from_numpy(BaseCoeff).to(device)
    #img = img.astype(np.float32)
    #img = np.moveaxis(img, -1, 0)

    for z in range(int(ratio / 2)):

        I1LRU = torch.zeros((bs, b, (2 ** (z + 1)) * r, (2 ** (z + 1)) * c), device=device, dtype=BaseCoeff.dtype)

        if z == 0:
            I1LRU[:, :, 1::2, 1::2] = img
        else:
            I1LRU[:, :, ::2, ::2] = img

        #I1LRU = np.expand_dims(I1LRU, axis=0)
        conv = nn.Conv2d(in_channels=b, out_channels=b, padding=(11, 0),
                         kernel_size=BaseCoeff.shape, groups=b, bias=False, padding_mode='circular')

        conv.weight.data = BaseCoeff
        conv.weight.requires_grad = False

        t = conv(torch.transpose(I1LRU, 2, 3))
        img = conv(torch.transpose(t, 2, 3))#.cpu().detach().numpy()
        #img = np.squeeze(img)

    #img = np.moveaxis(img, 0, -1)

    return img



def interp_3x_1d(img, N=50):
    ratio = 3

    bs, c, h, w = img.shape
    n = torch.arange(-N, N + 1)
    h1 = torch.sinc(n + 1 / ratio)
    h1 = h1 / torch.sum(h1)

    h1 = torch.fliplr(h1[None, :])
    h1 = h1[None, None, :, :]

    h2 = torch.sinc(n + 2 / ratio)
    h2 = h2 / torch.sum(h2)
    h2 = torch.fliplr(h2[None, :])
    h2 = h2[None, None, :, :]

    h1 = h1.repeat(c, 1, 1, 1).type(img.dtype).to(img.device)
    h2 = h2.repeat(c, 1, 1, 1).type(img.dtype).to(img.device)


    img_padded = pad(img, [N+1, 0, N , 0], padding_mode='symmetric')

    x1 = conv2d(img_padded, h1, padding='same', groups=c)
    x1 = x1[:, :, :, N+1:-N]

    x2 = conv2d(img_padded, h2, padding='same', groups=c)
    x2 = x2[:, :, :, N:-N-1]

    y = torch.zeros((bs, c, h, w * ratio), device=img.device, dtype=img.dtype)

    y[:, :, :, ::ratio] = x2
    y[:, :, :, 1::ratio] = img
    y[:, :, :, 2::ratio] = x1

    return y


def interp_3x_2d(img, N=30):

    z = interp_3x_1d(img, N)
    z = interp_3x_1d(z.transpose(2, 3), N)
    z = z.transpose(2, 3)
    return z


def ideal_interpolator(img, ratio):

    if ratio & (ratio - 1) == 0:
        img_upsampled = interp23tap_torch(img, ratio)
    else:
        img_upsampled = interp_3x_2d(img)
        img_upsampled = interp23tap_torch(img_upsampled, 2)

    return img_upsampled


def mtf_kernel_to_torch(h):
    """
        Compute the estimated MTF filter kernels for the supported satellites and calculate the spatial bias between
        each Multi-Spectral band and the Panchromatic (to implement the coregistration feature).
        Parameters
        ----------
        h : Numpy Array
            The filter based on Modulation Transfer Function.
        Return
        ------
        h : Tensor array
            The filter based on Modulation Transfer Function reshaped to Conv2d kernel format.
        """

    h = np.moveaxis(h, -1, 0)
    h = np.expand_dims(h, axis=1)
    h = h.astype(np.float32)
    h = torch.from_numpy(h).type(torch.float32)
    return h


def fsamp2(hd):
    """
        Compute fir filter with window method
        Parameters
        ----------
        hd : float
            Desired frequency response (2D)
        Return
        ------
        h : Numpy array
            The fir Filter
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = np.real(h)

    return h


def fir_filter_wind(f1, f2):
    """
        Compute fir filter with window method
        Parameters
        ----------
        f1 : float
            Desired frequency response (2D)
        f2 : Numpy Array
            The filter kernel (2D)
        Return
        ------
        h : Numpy array
            The fir Filter
    """

    hd = f1
    w1 = f2
    n = w1.shape[0]
    t = np.arange(start=-(n - 1) / 2, stop=(n - 1) / 2 + 1) * 2 / (n - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 ** 2 + t2 ** 2)

    dd = (t12 < t[0]) + (t12 > t[-1])

    t12[dd] = 0

    w = np.interp(t12.flatten(), t, w1).reshape(t12.shape)
    w[dd] = 0
    h = fsamp2(hd) * w

    return h


def fspecial_gauss(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function
        Parameters
        ----------
        size : Tuple
            The dimensions of the kernel. Dimension: H, W
        sigma : float
            The frequency of the gaussian filter
        Return
        ------
        h : Numpy array
            The Gaussian Filter of sigma frequency and size dimension
        """
    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def nyquist_filter_generator(nyquist_freq, ratio, kernel_size):
    """
        Compute the estimeted MTF filter kernels.
        Parameters
        ----------
        nyquist_freq : Numpy Array or List
            The MTF frequencies
        ratio : int
            The resolution scale which elapses between MS and PAN.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).
        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function.
    """
    assert isinstance(nyquist_freq, (np.ndarray, list)), 'Error: GNyq must be a list or a ndarray'
    if isinstance(nyquist_freq, list):
        nyquist_freq = np.asarray(nyquist_freq)
        nyquist_freq = np.reshape(nyquist_freq, (1, nyquist_freq.shape[0]))

    nbands = nyquist_freq.shape[1]

    kernel = np.zeros((kernel_size, kernel_size, nbands))  # generic kerenel (for normalization purpose)
    fcut = 1 / np.double(ratio)

    for j in range(nbands):
        alpha = np.sqrt(((kernel_size - 1) * (fcut / 2)) ** 2 / (-2 * np.log(nyquist_freq[0, j])))
        hh = fspecial_gauss((kernel_size, kernel_size), alpha)
        hd = hh / np.max(hh)
        h = np.kaiser(kernel_size, 0.5)
        h = np.real(fir_filter_wind(hd, h))

        h = np.real(h)
        kernel[:, :, j] = h

    return kernel

def gen_mtf(ratio, sensor='none', kernel_size=41, nbands=3):
    """
        Compute the estimated MTF filter kernels for the supported satellites.
        Parameters
        ----------
        ratio : int
            The resolution scale which elapses between MS and PAN.
        sensor : str
            The name of the satellites which has provided the images.
        kernel_size : int
            The size of the kernel (Only squared kernels have been implemented).
        Return
        ------
        kernel : Numpy array
            The filter based on Modulation Transfer Function for the desired satellite.
        """
    GNyq = []

    if sensor == 'S2-10':
        GNyq = [0.275, 0.28, 0.25, 0.24]
    elif sensor == 'S2-10-PAN':
        GNyq = [0.26125] * nbands
    elif sensor == 'S2-20':
        GNyq = [0.365, 0.33, 0.34, 0.32, 0.205, 0.235]
    elif sensor == 'S2-60':
        GNyq = [0.3175, 0.295]
    elif sensor == 'S2-60-1C':
        GNyq = [0.3175, 0.295, 0.30]
    elif sensor == 'WV3':
        GNyq = [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315] ## TO REMOVE
    else:
        GNyq = [0.3] * nbands

    h = nyquist_filter_generator(GNyq, ratio, kernel_size)

    return h

def mtf(img, sensor, ratio, mode='replicate'):
    h = gen_mtf(ratio, sensor, nbands=img.shape[1])

    h = mtf_kernel_to_torch(h).type(img.dtype).to(img.device)
    img_lp = conv2d(pad(img, (h.shape[-2] // 2, h.shape[-2] // 2, h.shape[-1] // 2, h.shape[-1] // 2), padding_mode=mode), h,
                    padding='valid', groups=img.shape[1])

    return img_lp