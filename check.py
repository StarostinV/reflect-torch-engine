import torch
import numpy as np

import matplotlib.pyplot as plt

from reflectivity import reflectivity
from test.fixtures.get_data import get_test_data


def main():
    for (test_path, slabs, data) in get_test_data():
        if data.shape[1] == 4:
            # only constant dq/q smearing is implemented so far.
            continue

        q = torch.from_numpy(data[:, 0])[:, None]
        thickness = torch.from_numpy(slabs[1:-1, 0])[None]
        sld = torch.from_numpy(slabs[:, 1] + 1j * slabs[:, 2])[None]
        roughness = torch.from_numpy(slabs[1:, 3])[None]

        t_data = reflectivity(
            q=q,
            thickness=thickness,
            sld=sld,
            roughness=roughness,
        ).squeeze().numpy()

        assert np.allclose(data[:, 1], t_data, rtol=8e-5)

        plt.semilogy(data[:, 0], data[:, 1])
        plt.semilogy(data[:, 0], t_data)
        plt.show()


if __name__ == '__main__':
    main()
