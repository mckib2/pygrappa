'''See if Cython module is working.'''

import numpy as np
from phantominator import shepp_logan

from cgrappa import cgrappa

# Generate fake sensitivity maps: mps
N = 512
ncoils = 4
xx = np.linspace(0, 1, N)
x, y = np.meshgrid(xx, xx)
mps = np.zeros((N, N, ncoils))
mps[..., 0] = x**2
mps[..., 1] = 1 - x**2
mps[..., 2] = y**2
mps[..., 3] = 1 - y**2

# generate 4 coil phantom
ph = shepp_logan(N)
imspace = ph[..., None]*mps
imspace = imspace.astype('complex')
ax = (0, 1)
kspace = 1/np.sqrt(N**2)*np.fft.fftshift(np.fft.fft2(
    np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

# crop 20x20 window from the center of k-space for calibration
pd = 10
ctr = int(N/2)
calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

# calibrate a kernel
kernel_size = (5, 5)

# undersample by a factor of 2 in both x and y
kspace[::2, 1::2, :] = 0
kspace[1::2, ::2, :] = 0

recon = cgrappa(kspace, calib, (5, 5))

from mr_utils import view
view(recon, fft=True)
