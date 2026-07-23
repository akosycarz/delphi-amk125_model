"""
Load a Delphi training configuration and command-line overrides.

Example:
    python train.py config/dryrun.py \
        --dataset=ukb_amk125_clinical_icd \
        --out_dir=models/dryrun_clinical_icd \
        --device=cuda \
        --dtype=bfloat16 \
        --compile=False

This file is executed from train.py with:
    exec(open("configurator.py").read())

It therefore updates train.py's global configuration variables directly.
"""

import os
import sys
from ast import literal_eval


def apply_override(key, raw_value):
    """Parse and apply one --key=value override."""
    if key not in globals():
        raise ValueError(f"Unknown config key: {key}")

    try:
        parsed_value = literal_eval(raw_value)
    except (SyntaxError, ValueError):
        parsed_value = raw_value

    current_value = globals()[key]

    if type(parsed_value) is not type(current_value):
        raise TypeError(
            f"Type mismatch for {key}: expected "
            f"{type(current_value).__name__}, got "
            f"{type(parsed_value).__name__}"
        )

    print(f"Overriding: {key} = {parsed_value}")
    globals()[key] = parsed_value


# Load config files first and apply --key=value arguments in command-line order.
for argument in sys.argv[1:]:
    if "=" not in argument:
        if argument.startswith("--"):
            raise ValueError(
                f"Invalid argument {argument!r}; expected --key=value"
            )

        config_file = os.path.expanduser(argument)

        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"Training config file not found: {config_file}"
            )

        print(f"Overriding config with {config_file}:")

        with open(config_file, encoding="utf-8") as handle:
            config_source = handle.read()

        print(config_source)
        exec(config_source, globals())
    else:
        if not argument.startswith("--"):
            raise ValueError(
                f"Invalid override {argument!r}; expected --key=value"
            )

        key, raw_value = argument[2:].split("=", 1)
        apply_override(key, raw_value)


# Load the vocabulary size and ignored-token list generated during
# preprocessing. The first path is the cluster layout; the second supports a
# data directory inside the current Delphi checkout.
dataset_name = globals().get("dataset")

if not isinstance(dataset_name, str) or not dataset_name:
    raise ValueError("The selected configuration did not define `dataset`")

dataset_value_candidates = [
    os.path.expanduser(
        os.path.join(
            "~/delphi-amk125_model/data",
            dataset_name,
            "config_values.py",
        )
    ),
    os.path.abspath(
        os.path.join(
            "data",
            dataset_name,
            "config_values.py",
        )
    ),
]

dataset_values_path = next(
    (
        path
        for path in dataset_value_candidates
        if os.path.isfile(path) and os.path.getsize(path) > 0
    ),
    None,
)

if dataset_values_path is None:
    searched_paths = "\n  - ".join(dataset_value_candidates)
    raise FileNotFoundError(
        "Could not find a non-empty config_values.py for dataset "
        f"{dataset_name!r}. Searched:\n  - {searched_paths}"
    )

print(f"Loading dataset values from {dataset_values_path}")

with open(dataset_values_path, encoding="utf-8") as handle:
    dataset_values_source = handle.read()

print(dataset_values_source)
exec(dataset_values_source, globals())
