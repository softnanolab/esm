# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch

import esm.pretrained
from esm.esmfold.v1.esmfold import ESMFold


def _load_model(model_name, weights_dir=None):
    """Load an ESMFold trunk, and with it the ESM-2 language model underneath.

    FORK CHANGE. `weights_dir` is not upstream: it says where weights fetched **by name**
    are read from and written to, instead of the one `torch.hub` cache every project on
    the machine shares. It reaches the ESM-2 download as well as this one -- which is the
    whole reason the argument exists, since an explicit path to the trunk alone still
    leaves 5.3 GB of language model in the shared cache. `None` keeps upstream's
    behaviour.

    `model_name` as a `.pt` path is unaffected: that file is named outright.
    """
    if model_name.endswith(".pt"):  # local, treat as filepath
        model_path = Path(model_name)
        model_data = torch.load(str(model_path), map_location="cpu")
    else:  # load from hub
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        model_data = torch.hub.load_state_dict_from_url(
            url,
            model_dir=esm.pretrained.weights_directory(weights_dir),
            progress=False,
            map_location="cpu",
        )

    cfg = model_data["cfg"]["model"]
    model_state = model_data["model"]
    model = ESMFold(esmfold_config=cfg, weights_dir=weights_dir)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    missing_essential_keys = []
    for missing_key in expected_keys - found_keys:
        if not missing_key.startswith("esm."):
            missing_essential_keys.append(missing_key)

    if missing_essential_keys:
        raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

    model.load_state_dict(model_state, strict=False)

    return model


def esmfold_v0(weights_dir=None):
    """
    ESMFold v0 model with 3B ESM-2, 48 folding blocks.
    This version was used for the paper (Lin et al, 2022). It was trained
    on all PDB chains until 2020-05, to ensure temporal holdout with CASP14
    and the CAMEO validation and test set reported there.
    """
    return _load_model("esmfold_3B_v0", weights_dir=weights_dir)


def esmfold_v1(weights_dir=None):
    """
    ESMFold v1 model using 3B ESM-2, 48 folding blocks.
    ESMFold provides fast high accuracy atomic level structure prediction
    directly from the individual sequence of a protein. ESMFold uses the ESM2
    protein language model to extract meaningful representations from the
    protein sequence.
    """
    return _load_model("esmfold_3B_v1", weights_dir=weights_dir)


def esmfold_structure_module_only_8M(weights_dir=None):
    """
    ESMFold baseline model using 8M ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 500K updates.
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_8M", weights_dir=weights_dir)


def esmfold_structure_module_only_8M_270K(weights_dir=None):
    """
    ESMFold baseline model using 8M ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 270K updates.
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_8M_270K", weights_dir=weights_dir)


def esmfold_structure_module_only_35M(weights_dir=None):
    """
    ESMFold baseline model using 35M ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 500K updates.
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_35M", weights_dir=weights_dir)


def esmfold_structure_module_only_35M_270K(weights_dir=None):
    """
    ESMFold baseline model using 35M ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 270K updates.
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_35M_270K", weights_dir=weights_dir)


def esmfold_structure_module_only_150M(weights_dir=None):
    """
    ESMFold baseline model using 150M ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 500K updates.
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_150M", weights_dir=weights_dir)


def esmfold_structure_module_only_150M_270K(weights_dir=None):
    """
    ESMFold baseline model using 150M ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 270K updates.
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_150M_270K", weights_dir=weights_dir)


def esmfold_structure_module_only_650M(weights_dir=None):
    """
    ESMFold baseline model using 650M ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 500K updates.
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_650M", weights_dir=weights_dir)


def esmfold_structure_module_only_650M_270K(weights_dir=None):
    """
    ESMFold baseline model using 650M ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 270K updates.
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_650M_270K", weights_dir=weights_dir)


def esmfold_structure_module_only_3B(weights_dir=None):
    """
    ESMFold baseline model using 3B ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 500K updates.
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_3B", weights_dir=weights_dir)


def esmfold_structure_module_only_3B_270K(weights_dir=None):
    """
    ESMFold baseline model using 3B ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 270K updates.
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_3B_270K", weights_dir=weights_dir)


def esmfold_structure_module_only_15B(weights_dir=None):
    """
    ESMFold baseline model using 15B ESM-2, 0 folding blocks.
    ESM-2 here is trained out to 270K updates.
    The 15B parameter ESM-2 was not trained out to 500K updates
    This is a model designed to test the capabilities of the language model
    when ablated for number of parameters in the language model.
    See table S1 in (Lin et al, 2022).
    """
    return _load_model("esmfold_structure_module_only_15B", weights_dir=weights_dir)
