import numpy as np
import torch

from reflectivity import reflectivity


def test_pytorch_engine(reflectivity_data):
    test_path, slabs, data = reflectivity_data

    if data.shape[1] == 4:
        # only constant dq/q smearing is implemented so far.
        return

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
