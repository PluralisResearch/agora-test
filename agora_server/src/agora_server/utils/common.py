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

import importlib

from functools import partial
from typing import Any

from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


def build_cls(class_name: str, init_args: dict | None = None, partial_init: bool = False) -> Any:
    """Instantiate class.

    Args:
        class_name (str): Full path to class.
        init_args (dict | None): Class init arguments. Defaults to None.
        partial_init (bool, optional): If True, return partial function. Defaults to False.

    Raises:
        Exception: Wrong class path/init arguments.

    Returns:
        Any: Class instance or partial function.
    """
    try:
        # Split module and class names
        module_name, class_name = class_name.rsplit(".", maxsplit=1)

        # Import module and get class
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)

        # Instantiate class
        if partial_init:
            instance = partial(class_, **init_args or {})
        else:
            instance = class_(**init_args or {})
        return instance
    except Exception as e:
        logger.error(f"Can't initialize class {class_name}: {e}", exc_info=logger.getEffectiveLevel() <= 15)
        raise
