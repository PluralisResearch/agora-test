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

import io

import boto3
import requests
import torch


def _split_location(ss_path: str, endpoint_url: str | None) -> tuple[str, str]:
    """(bucket, key) from a ``s3://``/``gs://`` URI or an endpoint URL (path or virtual-hosted)."""
    for scheme in ("s3://", "gs://"):
        if ss_path.startswith(scheme):
            bucket, _, key = ss_path[len(scheme) :].partition("/")
            return bucket, key
    if endpoint_url:
        if ss_path.startswith(endpoint_url):
            bucket, _, key = ss_path[len(endpoint_url) :].strip("/").partition("/")
            return bucket, key
        endpoint_host = endpoint_url.split("://", 1)[-1]
        host, _, key = ss_path.split("://", 1)[-1].partition("/")
        if host.endswith("." + endpoint_host):
            return host.removesuffix("." + endpoint_host), key
    raise RuntimeError(f"Cannot derive bucket/key from subspace component path {ss_path!r}")


def load_ss_components(
    ss_path: str,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    region_name: str | None = None,
    endpoint_url: str | None = None,
) -> dict:
    """Load rcv and fixed_embeddings.

    ``ss_path`` is either an object storage location — ``s3://...``, ``gs://...``, or an
    S3-compatible endpoint URL (e.g. Cloudflare R2) together with ``endpoint_url`` —
    fetched with an authenticated client, or any other HTTPS URL, fetched publicly.

    Args:
        ss_path (str): URL/URI to the subspace compression file.
        access_key_id (str | None, optional): Storage access key id. Defaults to None.
        secret_access_key (str | None, optional): Storage secret access key. Defaults to None.
        region_name (str | None, optional): Storage region. Defaults to None.
        endpoint_url (str | None, optional): Custom S3-compatible endpoint (e.g. GCS interop
            or Cloudflare R2). Defaults to None (AWS).

    Raises:
        RuntimeError: Failed to download.

    Returns:
        Dict of: rcv, fixed_tok_weight.
    """
    try:
        if ss_path.startswith(("s3://", "gs://")) or endpoint_url:
            bucket, key = _split_location(ss_path, endpoint_url)
            s3_client_kwargs = {}
            if region_name:
                s3_client_kwargs["region_name"] = region_name
            if access_key_id and secret_access_key:
                s3_client_kwargs["aws_access_key_id"] = access_key_id
                s3_client_kwargs["aws_secret_access_key"] = secret_access_key
            if endpoint_url:
                s3_client_kwargs["endpoint_url"] = endpoint_url

            s3_client = boto3.client("s3", **s3_client_kwargs)
            content = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
        else:
            response = requests.get(ss_path)
            response.raise_for_status()
            content = response.content

        # Load the state dict
        buffer = io.BytesIO(content)
        ss_comp_dict = torch.load(buffer, map_location="cpu", weights_only=True)
    except (requests.RequestException, RuntimeError) as e:
        raise RuntimeError(f"Failed to download subspace compression components: {e}") from e

    return ss_comp_dict
