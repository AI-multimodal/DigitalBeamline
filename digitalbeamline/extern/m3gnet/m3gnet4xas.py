"""M3GNet for Materials XAS!"""

from pathlib import Path

import numpy as np
import torch
from yaml import safe_load

from .featurizer import (
    featurize_material,
    _load_default_featurizer,
)


GRIDS = {
    "FEFF": {
        "Ti": np.linspace(4965, 5075, 200),
        "Cu": np.linspace(8983, 9124, 200),
    },
    "VASP": {
        "Ti": np.linspace(4715, 4765, 200)  # this is just a guess honestly
    }
}

ALLOWED_ABSORBERS = ["Ti"]

ALLOWED_XAS_TYPES = ["XANES"]

ALLOWED_THEORY = ["FEFF"]

ZOO_PATH = Path(__file__).parent.resolve() / "zoo"


def get_predictor(
    theory="FEFF",
    xas_type="XANES",
    version="230925",
    directory="Ti-O",
):
    """Returns a dictionary of the default configuration for predicting the
    XANES of materials.

    Parameters
    ----------
    theory : str, optional
        The level of theory at which the calculation was performed.
    xas_type : str, optional
        Either XANES or EXAFS.
    version : str, optional
        The version of the model. If there is a wildcard "*" in the version,
        it will interpret this as an ensemble and will attempt a match for all
        models of that type.
    directory : str, optional
        The directory in zoo in which the model is located. This allows us
        to resolve more precisely by training set (such as Ti-O vs Ti).

    Returns
    -------
    callable
        A function which takes a pymatgen.core.structure.Structure as input and
        returns a dictionary, resolved by site, of the predictions.
    """

    if xas_type not in ALLOWED_XAS_TYPES:
        raise NotImplementedError("Choose from xas_type in ['XANES'] only")

    if theory not in ALLOWED_THEORY:
        raise NotImplementedError("Only FEFF theory available right now")    

    # Currently this is all that's implemented
    def featurizer(structure):
        return featurize_material(structure, model=_load_default_featurizer())

    # Model signatures will be very specific
    model_signature = f"{theory}-{xas_type}-v{version}.pt"

    model_path = ZOO_PATH / Path(directory) / model_signature
    models = [torch.load(model_path)]

    metadata = safe_load(model_path / "metadata.yaml")
    absorber = metadata["absorber"]

    def predictor(structure, site_resolved=True):
        features = featurizer(structure)
        indexes = [
            ii
            for ii, site in enumerate(structure)
            if site.specie.symbol == absorber
        ]
        features = torch.tensor(features[indexes, :])
        preds = np.array(
            [m(features).detach().numpy() for m in models]
        ).swapaxes(0, 1)

        return {
            site: pred.squeeze() for site, pred in zip(indexes, preds)
        }

    return predictor
