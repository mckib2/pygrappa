'''Python implementation of Split-Slice-GRAPPA.'''

from pygrappa import slicegrappa


def splitslicegrappa(*args, **kwargs):
    '''Split-Slice-GRAPPA.

    Notes
    -----
    This is an alias for pygrappa.slicegrappa(split=True).
    See pygrappa.slicegrappa() for more information.
    '''

    # Make sure that the 'split' argument is set to True
    if 'split' not in kwargs or not kwargs['split']:
        kwargs['split'] = True
    return slicegrappa(*args, **kwargs)


if __name__ == '__main__':
    pass
