from pathlib import Path
import numpy as np

unpolarized_dir = Path(__file__).parents[1] / 'unpolarised'


def get_test_data():
    """
    A generator yielding (slabs, data) tuples.

    `slabs` are np.ndarrays that specify the layer structure of the test.
    ``slabs.shape = (N + 2, 4)``, where N is the number of layers.

    The layer specification file has the layout:

    ignored     SLD_fronting ignored      ignored
    thickness_1 SLD_1        iSLD_1       rough_fronting1
    thickness_2 SLD_2        iSLD_2       rough_12
    ignored     SLD_backing  iSLD_backing rough_backing2

    `data` contains the test reflectivity data. It's an np.ndarray with
    shape `(M, N)`, where M is the number of datapoints, and N in [2, 3, 4].
    The first column contains Q points (reciprocal Angstrom), the second column
    the reflectivity data, the optional third and fourth columns are dR and dQ.
    If present dR and dQ are 1 standard deviation uncertainties on reflectivity
    and Q-resolution (gaussian resolution kernel).
    """

    test_paths = unpolarized_dir.glob('*.txt')

    for test_path in test_paths:
        # layers/data tuples
        layers_file, data_file = get_data(test_path)

        slabs = np.loadtxt(unpolarized_dir / layers_file)
        assert slabs.shape[1] == 4

        data = np.loadtxt(unpolarized_dir / data_file)
        assert data.shape[1] in [2, 3, 4]

        yield test_path, slabs, data


def get_data(path):
    # for each of the test files extract the parameters,
    # e.g. the names of the data_file and layers_file

    with open(path, "rt") as f:
        # ignore comment lines starting with #
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                yield line
