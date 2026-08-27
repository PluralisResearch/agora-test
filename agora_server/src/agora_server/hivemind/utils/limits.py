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

from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


def increase_file_limit(new_soft=2**15, new_hard=2**15):
    """Increase the maximum number of open files. On Linux, this allows spawning more processes/threads."""
    try:
        import resource  # local import to avoid ImportError for Windows users

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = max(soft, new_soft)
        new_hard = max(hard, new_hard)
        logger.info(f"Increasing file limit: soft {soft}=>{new_soft}, hard {hard}=>{new_hard}")
        return resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, new_hard))
    except Exception as e:
        logger.warning(f"Failed to increase file limit: {e}")
