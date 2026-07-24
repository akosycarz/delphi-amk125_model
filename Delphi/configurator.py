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
import shutil
import subprocess
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
# preprocessing from the same data root and dataset that train.py will use.
dataset_name = globals().get("dataset")
configured_data_root = globals().get("data_root")

if not isinstance(dataset_name, str) or not dataset_name:
    raise ValueError("The selected configuration did not define `dataset`")
if not isinstance(configured_data_root, str) or not configured_data_root:
    raise ValueError("The selected configuration did not define `data_root`")

dataset_dir = os.path.expanduser(
    os.path.join(configured_data_root, dataset_name)
)
python_values_path = os.path.join(dataset_dir, "config_values.py")
rds_values_path = os.path.join(dataset_dir, "vocab_meta.rds")

if os.path.isfile(python_values_path) and os.path.getsize(python_values_path) > 0:
    dataset_values_path = python_values_path
    print(f"Loading dataset values from {dataset_values_path}")

    with open(dataset_values_path, encoding="utf-8") as handle:
        dataset_values_source = handle.read()

    print(dataset_values_source)
    exec(dataset_values_source, globals())
elif os.path.isfile(rds_values_path) and os.path.getsize(rds_values_path) > 0:
    dataset_values_path = rds_values_path
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            f"Found {rds_values_path}, but Rscript is not installed or is not "
            "on PATH. Install R or generate config_values.py with "
            "scripts/write_ukb_amk125_config_values.R."
        )

    # vocab_meta.rds is an R list containing vocab_size and ignore_tokens.
    # Emit two simple lines so no third-party Python RDS reader is required.
    r_source = (
        "args <- commandArgs(trailingOnly=TRUE); "
        "meta <- readRDS(args[[1]]); "
        "if (is.null(meta$vocab_size) || is.null(meta$ignore_tokens)) "
        "stop('RDS must contain vocab_size and ignore_tokens'); "
        "cat(as.integer(meta$vocab_size), '\\n'); "
        "cat(paste(as.integer(meta$ignore_tokens), collapse=','), '\\n')"
    )
    result = subprocess.run(
        [rscript, "-e", r_source, rds_values_path],
        check=True,
        capture_output=True,
        text=True,
    )
    output_lines = result.stdout.splitlines()
    if len(output_lines) < 2:
        raise ValueError(
            f"Could not read vocab_size and ignore_tokens from {rds_values_path}"
        )

    vocab_size = int(output_lines[0].strip())
    ignore_tokens = [
        int(token)
        for token in output_lines[1].split(",")
        if token.strip()
    ]
    globals()["vocab_size"] = vocab_size
    globals()["ignore_tokens"] = ignore_tokens
    print(
        f"Loaded dataset values from {dataset_values_path}: "
        f"vocab_size={vocab_size}, ignore_tokens={ignore_tokens}"
    )
else:
    raise FileNotFoundError(
        f"Could not find dataset vocabulary metadata for {dataset_name!r}. "
        f"Expected either:\n  - {python_values_path}\n  - {rds_values_path}"
    )

if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
    raise ValueError(
        f"{dataset_values_path} must define vocab_size as a positive integer"
    )
if not isinstance(ignore_tokens, list) or not all(
    isinstance(token, int) and not isinstance(token, bool)
    for token in ignore_tokens
):
    raise ValueError(
        f"{dataset_values_path} must define ignore_tokens as a list of integers"
    )
invalid_ignore_tokens = [
    token for token in ignore_tokens if token < 0 or token >= vocab_size
]
if invalid_ignore_tokens:
    raise ValueError(
        f"{dataset_values_path} has ignore_tokens outside the valid range "
        f"0..{vocab_size - 1}: {invalid_ignore_tokens}"
    )
