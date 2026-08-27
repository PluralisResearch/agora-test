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

import copy
import importlib.util

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from agora_server.hivemind.utils.logging import get_logger
from agora_server.models.base_arguments import ModelArguments


logger = get_logger(__name__)


class ModelArgumentsRegistry:
    """Global registry for all model arguments.

    This class allows dynamic registration and retrieval of model arguments
    for different models. It supports lazy loading of model-specific argument
    modules to avoid unnecessary imports.

    **Example:**
    >>> from agora_server.models.arguments_registry import ModelArgumentsRegistry
    >>> from agora_server.models.base_arguments import ModelArguments

    >>> class LlamaArguments(ModelArguments):
    >>>     ... # implementation of the arguments

    >>> ModelArgumentsRegistry.register("llama", "debugmodel_small", LlamaArguments(hidden_dim=256))
    >>> model_args = ModelArgumentsRegistry.get("llama", "debugmodel_small")

    >>> # To override specific arguments
    >>> model_args_overridden = ModelArgumentsRegistry.get("llama", "debugmodel_small", hidden_dim=512)

    >>> # Load custom models from a file
    >>> ModelArgumentsRegistry.add_custom_arguments("/path/to/custom_arguments.py")
    """

    _registry: dict[str, dict[str, ModelArguments]] = {}
    _loaded_models: set[str] = set()

    @classmethod
    def register(cls, model: str, config_name: str, args: ModelArguments) -> None:
        """Register a config under a model type."""
        if model not in cls._registry:
            cls._registry[model] = {}
            cls._loaded_models.add(model)  # Mark as loaded since we're registering directly

        if config_name in cls._registry[model]:
            logger.warning(
                f"Config with name {config_name} for model {model} is already registered. This might be expected if the module is re-imported."
            )

        cls._registry[model][config_name] = args

    @classmethod
    def _ensure_model_loaded(cls, model: str):
        """Dynamically import the model's arguments module to trigger registration."""
        if model in cls._loaded_models:
            return

        try:
            # Dynamically import: agora_server.models.{model}.arguments
            module_path = f"agora_server.models.{model}.arguments"
            importlib.import_module(module_path)
            cls._loaded_models.add(model)
        except ImportError as e:
            raise RuntimeError(f"Failed to load arguments for model '{model}': {e}") from None

    @classmethod
    def get(cls, model: str, config_name: str, **overrides) -> ModelArguments:
        """Get a config with optional overrides.

        **Note:**
            llama parameter `replace_n_layers` needs to be set up before `expert_name` otherwise the number of total layers will be falsely overwritten.
            Correct usage: a = ModelArgumentsRegistry.get("llama", "500M_C_test", replace_n_layers=False, expert_name="lm_head")

        Args:
            model (str): The model name, e.g. "llama".
            config_name (str): The config name, e.g. "debugmodel_C".
            **overrides: Key-value pairs to override specific arguments in the config.

        Raises:
            KeyError: If the model type is unknown.
            KeyError: If the config name is unknown.

        Returns:
            ModelArguments: The requested model arguments, possibly overridden.
        """
        cls._ensure_model_loaded(model)

        if model not in cls._registry:
            available_types = ", ".join(cls._registry.keys())
            raise KeyError(f"Unknown model type: '{model}'. Available types: {available_types}")

        if config_name not in cls._registry[model]:
            available = ", ".join(cls._registry[model].keys())
            raise KeyError(f"Unknown {model} config: '{config_name}'. Available: {available}")

        config = cls._registry[model][config_name]

        if overrides:
            if hasattr(config, "__dataclass_fields__"):
                return replace(config, **overrides)
            else:
                config = copy.deepcopy(config)
                for key, value in overrides.items():
                    setattr(config, key, value)
        return config

    @classmethod
    def list_configs(cls) -> dict[str, list[str]]:
        """List all configs."""
        return {model: list(configs.keys()) for model, configs in cls._registry.items()}

    @staticmethod
    def add_custom_arguments(path: str | Sequence[str]):
        """Load custom model arguments from given file path(s)."""
        if isinstance(path, str):
            path = [path]

        for p in path:
            p = Path(p)
            spec = importlib.util.spec_from_file_location(
                name=f"custom_args_module_{p.stem.replace('-', '_')}",
                location=p.resolve().as_posix(),
            )

            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load custom model arguments from path: {p}")

            try:
                custom_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_module)
            except Exception as e:
                raise ImportError(f"Failed to import custom model arguments from {p}: {e}") from None
