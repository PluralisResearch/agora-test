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

from __future__ import annotations

import re

from typing import NamedTuple, Tuple, Union

from agora_server.hivemind.p2p import PeerID


ExpertUID, ExpertPrefix, Coordinate = str, str, int


class ExpertInfo(NamedTuple):
    uid: ExpertUID
    peer_id: PeerID


UID_DELIMITER = "."  # when declaring experts, DHT store all prefixes of that expert's uid, split over this prefix
FLAT_EXPERT = -1  # grid prefix reserved for storing 1d expert uids. Used to speed up find_best_experts in 1d case.
UID_PATTERN = re.compile("^(([^.])+)([.](?:[0]|([1-9]([0-9]*))))+$")  # e.g. ffn_expert.98.76.54 - prefix + some dims


def is_valid_uid(maybe_uid: str) -> bool:
    """An uid must contain a string expert type, followed by one or more .-separated numeric indices"""
    return bool(UID_PATTERN.fullmatch(maybe_uid))


def split_uid(uid_or_prefix: ExpertUID | ExpertPrefix) -> tuple[ExpertPrefix, Coordinate]:
    """Separate an expert UID or prefix into a new ExpertPrefix and integer for the last coordinate"""
    uid_or_prefix = uid_or_prefix.rstrip(UID_DELIMITER)
    pivot = uid_or_prefix.rindex(UID_DELIMITER) + 1
    return uid_or_prefix[:pivot], int(uid_or_prefix[pivot:])
