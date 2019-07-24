'''See if Cython module is working.'''

# import numpy as np
# from pyx_cgrappa import multiply_by_10 # pylint: disable=E0611
#
# a = np.ones(5, dtype=np.double)
# print(multiply_by_10(a))
#
# b = np.ones(10, dtype=np.double)
# b = b[::2]  # b is not contiguous.
#

# import numpy as np

# kspace = np.arange((5**3))
# kspace = np.reshape(kspace, (5, 5, 5))
# kspace = kspace + 1j*kspace
#
# # kspace = np.ones((5, 5, 5), dtype='complex')
# calib = np.ones((3, 3, 5), dtype='complex')
#
# cgrappa(kspace, calib, (2, 2))

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
