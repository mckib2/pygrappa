'''Do kSPA using BART stuff.'''

# from time import time

# import numpy as np
# import matplotlib.pyplot as plt
# from bart import bart  # pylint: disable=E0401

# from pygrappa import kspa
# from utils import gridder

if __name__ == '__main__':
    pass
    # # raise NotImplementedError('kSPA not ready yet, sorry...')
    #
    # sx, spokes, nc = 16, 16, 4
    # traj = bart(1, 'traj -r -x%d -y%d' % (sx, spokes))
    # kx, ky = traj[0, ...].real.flatten(), traj[1, ...].real.flatten()
    #
    # # Use BART to get Shepp-Logan and sensitivity maps
    # t0 = time()
    # k = bart(1, 'phantom -k -s%d -t' % nc, traj).reshape((-1, nc))
    # print('Took %g seconds to simulate %d coils' % (time() - t0, nc))
    # sens = bart(1, 'phantom -S%d -x%d' % (nc, sx)).squeeze()
    # # ksens = bart(1, 'fft -u 3', sens)
    #
    # # Undersample
    # ku = k.copy()
    # # ku[::4] = 0
    #
    # # Reconstruct using kSPA
    # res = kspa(kx, ky, ku, sens)
    # # assert False
    # # fil = np.hamming(sx)[:, None]*np.hamming(sx)[None, :]
    # # res = res*fil
    # plt.imshow(np.abs(res))
    # plt.show()
    #
    # # Take a looksie
    # sos = lambda x0: np.sqrt(np.sum(np.abs(x0)**2, axis=-1))
    # ifft = lambda x0: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
    #     x0)))
    # plt.subplot(1, 3, 1)
    # plt.imshow(sos(gridder(kx, ky, k, sx, sx)))
    # plt.title('Truth')
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(sos(gridder(kx, ky, ku, sx, sx)))
    # plt.title('Undersampled')
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(np.abs(ifft(res)))
    # plt.title('kSPA')
    #
    # plt.show()
