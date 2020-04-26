
import numpy as np


def findRectangle2d(region, mask, ctr):
    # Push out on each face, backtracking where necessary
    acs = np.zeros(region.shape, dtype=bool)
    acs[ctr] = True

    # go up
    for ii in range(ctr[0], 0, -1):
        top = ii
        if mask[ii, ctr[1]]:
            acs[ii, ctr[1]] = True
        else:
            break
    # go down
    for ii in range(ctr[0], acs.shape[0]):
        bottom = ii+1
        if mask[ii, ctr[1]]:
            acs[ii, ctr[1]] = True
        else:
            break
    # push right
    for jj in range(ctr[1], acs.shape[1]):
        if all(mask[top:bottom, jj]):
            acs[top:bottom, jj] = True
        else:
            # Trim False entries
            if not mask[top, jj]:
                top += 1
            if not mask[bottom-1, jj]:
                bottom -= 1
            if all(mask[top:bottom, jj]):
                acs[top-1, :] = False
                acs[bottom, :] = False
                acs[top:bottom, jj] = True
            else:
                top -= 1
                bottom += 1
                break
    # push left
    for jj in range(ctr[1], 0, -1):
        if all(mask[top:bottom, jj]):
            acs[top:bottom, jj] = True
        else:
            # Trim False entries
            if not mask[top, jj]:
                top += 1
            if not mask[bottom-1, jj]:
                bottom -= 1
            if all(mask[top:bottom, jj]):
                acs[top-1, :] = False
                acs[bottom, :] = False
                acs[top:bottom, jj] = True
            else:
                top -= 1
                bottom += 1
                break

    return(acs, top, bottom)
