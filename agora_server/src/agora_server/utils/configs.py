# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


OmegaConf.register_new_resolver("sum", lambda *numbers: sum(numbers))
OmegaConf.register_new_resolver("mul", lambda *numbers: math.prod(numbers))


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML config and recursively merge all included configs.

    Supports `_override_: true` marker for full replacement instead of merging.
    """
    # Load the main config
    try:
        config = OmegaConf.load(config_path)
        if not isinstance(config, DictConfig):
            raise ValueError(f"Config at {config_path} can't be loaded as a DictConfig.")
    except Exception as e:
        raise RuntimeError(f"Failed to load config at {config_path}: {e}") from None

    # Check if there are includes
    if "include" in config:
        includes = config.pop("include")

        if not isinstance(includes, ListConfig):
            includes = [includes]

        # Start with an empty config to merge into
        merged_config = OmegaConf.create({})

        # Process each include
        config_dir = Path(config_path).parent
        for include_path in includes:
            if not Path(include_path).is_absolute():
                # Resolve the include path relative to the current config
                full_include_path = (config_dir / include_path).resolve()
            else:
                full_include_path = Path(include_path)

            if not full_include_path.exists():
                raise FileNotFoundError(
                    f"Error while parsing {config_path} file. Included config not found: {full_include_path}"
                )

            # Recursively load the included config
            included_config = load_config(full_include_path)

            # Merge the included config using custom merge
            merged_config = custom_merge(merged_config, included_config)

        # Merge the current config on top
        merged_config = custom_merge(merged_config, config)
    else:
        merged_config = config

    return merged_config


def custom_merge(base: DictConfig, override: DictConfig) -> DictConfig:
    """Custom merge function that respects the `_override_` marker.

    If a dictionary in the override config contains the key `_override_`, it replaces
    the corresponding dictionary in the base config entirely instead of merging.
    """
    base_dict = OmegaConf.to_container(base, resolve=False)
    override_dict = OmegaConf.to_container(override, resolve=False)

    def merge_recursive(base_obj, override_obj):
        if not isinstance(override_obj, dict):
            return override_obj

        if not isinstance(base_obj, dict):
            return override_obj

        result = base_obj.copy()

        for key, override_value in override_obj.items():
            if isinstance(override_value, dict) and "_override_" in override_value:
                # If override has _override_ marker, replace entirely instead of merging
                cleaned_value = {k: v for k, v in override_value.items() if k != "_override_"}
                result[key] = cleaned_value
            elif key in result and isinstance(result[key], dict) and isinstance(override_value, dict):
                # Recursive merge for nested dicts
                result[key] = merge_recursive(result[key], override_value)
            else:
                # Direct override
                result[key] = override_value

        return result

    merged_dict = merge_recursive(base_dict, override_dict)
    return OmegaConf.create(merged_dict)


def validate_config(config: DictConfig | ListConfig) -> None:
    """Validate that the config does not contain any missing mandatory values."""
    missing = OmegaConf.missing_keys(config)
    if missing:
        missing_list = "; ".join(sorted(missing))
        raise ValueError(f"Config contains missing mandatory values (???): {missing_list}")
