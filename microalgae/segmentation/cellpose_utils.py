from typing import Optional

try:
    from cellpose import models
except Exception as e:
    # Defer import-time errors to call time; keeps module importable in environments without cellpose
    models = None  # type: ignore


def load_cellpose_model(model_folder_or_pretrained: str = "cellpose", gpu: bool = True):
    """
    Load Cellpose model in a unified way.

    Parameters
    - model_folder_or_pretrained: path to a pretrained model or a model identifier understood by Cellpose
    - gpu: whether to attempt using GPU; will be disabled automatically if unavailable
    """
    if models is None:
        raise ImportError("cellpose is not installed or failed to import; cannot load Cellpose model")

    use_gpu = gpu and (models.use_gpu() is not None)
    model = models.CellposeModel(pretrained_model=model_folder_or_pretrained, gpu=use_gpu)
    return model
