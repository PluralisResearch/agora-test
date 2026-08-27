# This file contains code originally from Hivemind under MIT License
# Original: Copyright 2020 Learning@home authors and collaborators
# Modified by: Pluralis Research 2026
#
# Original code: MIT License (see THIRD_PARTY_LICENSES)
# Modifications: Apache 2.0 License (see LICENSE)
#
# Licensed under the Apache License, Version 2.0 (the "License") for modifications only;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

# connection_handler is imported here so the lazy `moe.server.connection_handler` attribute
# lookup in moe.client.expert.get_server_stub resolves.
from . import connection_handler  # noqa: F401
from .module_backend import ModuleBackend


__all__ = ["ModuleBackend", "connection_handler"]
