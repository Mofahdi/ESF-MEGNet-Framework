"""Shared experiment defaults and validation for the Python workflow runners."""

MODELS = ("MEGNet", "ESF_MEGNet", "ESF_MEGNet_zero", "ESF_MEGNet_rand")
DEFAULT_SETTINGS = {
    "seed": 42, "feature_seed": 42, "split_seed": 42,
    "n_splits": 10, "max_epochs": 500, "patience": 300,
    "batch_size": 64, "decay_steps": 1000, "device": "cuda",
    "models": list(MODELS),
}


def validate_settings(supplied):
    if not isinstance(supplied, dict) or set(supplied) - DEFAULT_SETTINGS.keys():
        raise ValueError(f"Unknown settings; allowed keys: {', '.join(DEFAULT_SETTINGS)}")
    settings = {**DEFAULT_SETTINGS, **supplied}
    for name in ("seed", "feature_seed", "split_seed", "n_splits", "max_epochs", "patience", "batch_size", "decay_steps"):
        value = settings[name]
        if type(value) is not int:
            raise ValueError(f"Setting {name} must be an integer")
        minimum = 0 if name in ("seed", "feature_seed", "split_seed", "patience") else 1
        if value < minimum or (name.endswith("seed") and value >= 2**32):
            raise ValueError(f"Setting {name} is outside its valid range")
    if settings["n_splits"] < 2:
        raise ValueError("n_splits must be at least 2")
    if settings["device"] not in ("cpu", "cuda", "gpu", "auto"):
        raise ValueError("device must be cpu, cuda, gpu, or auto")
    models = settings["models"]
    if not isinstance(models, list) or not models or any(model not in MODELS for model in models):
        raise ValueError(f"models must be a nonempty list selected from {MODELS}")
    if len(set(models)) != len(models):
        raise ValueError("models contains duplicate entries")
    settings["models"] = list(models)
    return settings
