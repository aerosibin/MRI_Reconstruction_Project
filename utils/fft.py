import numpy as np


def fft2c(img):
    """Centered 2D FFT with orthogonal normalization.

    img: HxW real or complex
    returns: HxW complex k-space (fftshifted)
    """
    k = np.fft.fft2(img, norm="ortho")
    k = np.fft.fftshift(k)
    return k


def ifft2c(k):
    """Inverse centered 2D FFT (ifftshift + ifft2) with orthogonal normalization.

    k: HxW complex k-space (fftshifted)
    returns: HxW complex image
    """
    k = np.fft.ifftshift(k)
    img = np.fft.ifft2(k, norm="ortho")
    return img
