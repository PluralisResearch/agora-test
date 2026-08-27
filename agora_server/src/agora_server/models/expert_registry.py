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

import importlib.util

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn as nn

from agora_server.hivemind.utils.logging import get_logger


SampleInputFunc = Callable[..., Sequence[torch.Tensor] | torch.Tensor]
logger = get_logger(__name__)


class ExpertRegistry:
    """Registry for expert classes and their sample inputs.

    This class allows dynamic registration and retrieval of expert classes
    for different models. It supports lazy loading of model-specific expert
    modules to avoid unnecessary imports.

    **Example:**
    >>> @ExpertRegistry.register("llama", "head", head_sample_input)
    >>> class HeadExpert(nn.Module):
    >>>     ... # implementation of the expert

    >>> expert_class = ExpertRegistry.get_expert_class("llama", "head")
    >>> sample_input_func = ExpertRegistry.get_sample_input("llama", "head")
    >>> (expert_class, sample_input_func) = ExpertRegistry.get_expert_info("llama", "head")

    >>> # Load custom models from a file
    >>> ExpertRegistry.add_custom_models("/path/to/custom_experts.py")
    """

    _registry: dict[str, dict[str, tuple[type[nn.Module], SampleInputFunc]]] = {}
    _loaded_models: set[str] = set()

    @classmethod
    def register(cls, model: str, expert_name: str, sample_input: SampleInputFunc):
        def decorator(expert_class: type[nn.Module]):
            if model not in cls._registry:
                cls._registry[model] = {}
                cls._loaded_models.add(model)  # Mark as loaded since we're registering directly

            if expert_name in cls._registry[model]:
                logger.warning(
                    f"Expert class with name {expert_name} for model {model} is already registered. This might be expected if the module is re-imported."
                )

            cls._registry[model][expert_name] = (expert_class, sample_input)
            return expert_class

        return decorator

    @classmethod
    def _ensure_model_loaded(cls, model: str):
        """Dynamically import the model's experts module to trigger registration."""
        if model in cls._loaded_models:
            return

        try:
            # Dynamically import: agora_server.models.{model}.experts
            module_path = f"agora_server.models.{model}.experts"
            importlib.import_module(module_path)
            cls._loaded_models.add(model)
        except ImportError as e:
            raise RuntimeError(f"Failed to load experts for model '{model}': {e}") from None

    @classmethod
    def get_expert_class(cls, model: str, expert_name: str) -> type[nn.Module]:
        """Get the expert class for a specific model and expert name.

        Args:
            model (str): The model name, e.g. "llama".
            expert_name (str): The expert name, e.g. "head".

        Raises:
            RuntimeError: If the expert class is not registered.

        Returns:
            type[nn.Module]: The expert class.
        """
        cls._ensure_model_loaded(model)
        model_to_expert_dict = cls._registry.get(model, {})
        expert_info = model_to_expert_dict.get(expert_name)
        if expert_info is None:
            raise RuntimeError(f"Expert class with name {expert_name} for model {model} is not registered.")
        return expert_info[0]

    @classmethod
    def get_sample_input(cls, model: str, expert_name: str) -> SampleInputFunc:
        """Get the sample input for a specific model and expert name.

        Args:
            model (str): The model name, e.g. "llama".
            expert_name (str): The expert name, e.g. "head".

        Raises:
            RuntimeError: If the expert class is not registered.

        Returns:
            SampleInputFunc: The sample input function.
        """
        cls._ensure_model_loaded(model)
        model_to_expert_dict = cls._registry.get(model, {})
        expert_info = model_to_expert_dict.get(expert_name)
        if expert_info is None:
            raise RuntimeError(f"Expert class with name {expert_name} for model {model} is not registered.")
        return expert_info[1]

    @classmethod
    def get_expert_info(cls, model: str, expert_name: str) -> tuple[type[nn.Module], SampleInputFunc]:
        """Get the expert class and sample input function for a specific model and expert name.

        Args:
            model (str): The model name, e.g. "llama".
            expert_name (str): The expert name, e.g. "head".

        Raises:
            RuntimeError: If the expert class is not registered.

        Returns:
            tuple[type[nn.Module], SampleInputFunc]: The expert class and sample input function.
        """
        cls._ensure_model_loaded(model)
        model_to_expert_dict = cls._registry.get(model, {})
        expert_info = model_to_expert_dict.get(expert_name)
        if expert_info is None:
            raise RuntimeError(f"Expert class with name {expert_name} for model {model} is not registered.")
        return expert_info

    @classmethod
    def list_experts(cls) -> dict[str, list[str]]:
        """List all registered experts."""
        return {model: list(experts.keys()) for model, experts in cls._registry.items()}

    @staticmethod
    def add_custom_models(path: str | Sequence[str]):
        """Load custom model experts from given file path(s)."""
        if isinstance(path, str):
            path = [path]

        for p in path:
            p = Path(p)
            spec = importlib.util.spec_from_file_location(
                name=f"custom_module_{p.stem.replace('-', '_')}",
                location=p.resolve().as_posix(),
            )

            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load custom model experts from path: {p}")

            try:
                custom_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_module)
            except Exception as e:
                raise ImportError(f"Failed to import custom model experts from {p}: {e}") from None
