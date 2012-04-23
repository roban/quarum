import h5py

def save_hdf5(filename, dict_like):
    """Save a dictionary-like object to an HDF5 file.

    Each key, value pair is stored in the HDF5 file.
    """
    f = h5py.File(filename, 'w')
    for k, v in dict_like.iteritems():
        f[k] = v
    f.close()

def load_hdf5(filename):
    """Read an HDF file and return a dictionary.
    """
    f = h5py.File(filename, 'r')
    data = dict()
    for k, v in f.iteritems():
        if len(v.shape) == 0:
            data[k] = v.value
        else:
            data[k] = v[:]
    f.close()
    return data
