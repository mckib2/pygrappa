'''Simple g-factor maps.'''

import numpy as np
import matplotlib.pyplot as plt

from pygrappa import gfactor, gfactor_single_coil_R2
from pygrappa.utils import gaussian_csm

if __name__ == '__main__':

    # Make circle
    N, nc = 128, 8
    X, Y = np.meshgrid(
        np.linspace(-1, 1, N),
        np.linspace(-1, 1, N))
    ph = X**2 + Y**2 < .9**2

    # Try single coil, R=2. For single coil, we'll need to add
    # background phase variation so we can pull pixels apart
    _, phi = np.meshgrid(
        np.linspace(0, np.pi, N),
        np.linspace(0, np.pi, N))
    phi = np.exp(1j*phi)
    Rx, Ry = 2, 1
    g_c1_R2_analytical = gfactor_single_coil_R2(ph*phi, Rx=Rx, Ry=Ry)
    g_c1_R2 = gfactor((ph*phi)[..., None], Rx=Rx, Ry=Ry)

    # Try multicoil
    coils = ph[..., None]*gaussian_csm(N, N, nc)
    Rx, Ry = 1, 3
    g_c8_R3 = gfactor(coils, Rx=Rx, Ry=Ry)

    # Let's take a look
    nx, ny = 1, 3
    plt_args = {
        'vmin': 0,
        'vmax': np.max(np.concatenate(
            (g_c1_R2_analytical, g_c1_R2)).flatten())
    }
    plt.subplot(nx, ny, 1)
    plt.imshow(g_c1_R2_analytical, **plt_args)
    plt.title('Single coil, Rx=2, Analytical')

    plt.subplot(nx, ny, 2)
    plt.imshow(g_c1_R2, **plt_args)
    plt.title('Single coil, Rx=2')

    plt.subplot(nx, ny, 3)
    plt.imshow(g_c8_R3)
    plt.title('%d coil, Rx/Ry=%d/%d' % (nc, Rx, Ry))

    plt.show()
