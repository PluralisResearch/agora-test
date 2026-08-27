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


import os
import threading
import time

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
import torch

from agora_server.hivemind.moe.server.module_backend import ModuleBackend
from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)

_CHUNK_RETRIES = 3
_CONNECT_TIMEOUT = 10  # seconds
_READ_TIMEOUT = 60  # seconds
_READ_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB
_SPEED_WINDOW = 30.0  # seconds of history used for speed/ETA calculation


class StateDownloader:
    """Fast parallel downloader using HTTP Range requests.
    Downloads directly into the final file at the correct byte offsets.
    """

    def __init__(self, url: str, download_threads: int = 8, gpu_id: int = 0):
        self.url = url
        self.dl_threads = download_threads
        self.gpu_id = gpu_id

        self._dest: str | None = None
        self._total_size: int = 0
        self._bytes_downloaded: int = 0
        self._lock = threading.Lock()
        self._speed_samples: deque[tuple[float, int]] = deque()  # (timestamp, cumulative bytes)
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._finished = threading.Event()
        self._cancelled = threading.Event()
        self._success: bool = False
        self._errors: list[Exception] = []
        self._thread: threading.Thread | None = None

    def start(self) -> "StateDownloader":
        """Start the download in a background thread."""
        logger.info(f"Starting state download with {self.dl_threads} threads...")
        self._thread = threading.Thread(target=self._run, daemon=True, name="state-downloader")
        self._thread.start()
        return self

    def get_progress(self) -> float:
        if self._total_size == 0:
            return 0.0
        return min(self._bytes_downloaded / self._total_size, 1.0)

    def get_eta(self) -> float:
        with self._lock:
            if len(self._speed_samples) < 2 or self._total_size == 0:
                return -1.0
            oldest_t, oldest_b = self._speed_samples[0]
            newest_t, newest_b = self._speed_samples[-1]
            elapsed = newest_t - oldest_t
            if elapsed == 0:
                return -1.0
            speed = (newest_b - oldest_b) / elapsed
            if speed == 0:
                return -1.0
            return (self._total_size - newest_b) / speed

    def get_speed(self) -> int:
        with self._lock:
            if len(self._speed_samples) < 2:
                return 0
            oldest_t, oldest_b = self._speed_samples[0]
            newest_t, newest_b = self._speed_samples[-1]
            elapsed = newest_t - oldest_t
            if elapsed == 0:
                return 0
            return int((newest_b - oldest_b) / elapsed)

    def is_finished(self) -> bool:
        return self._finished.is_set()

    def get_dl_time(self) -> float:
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time is not None else time.time()
        return end - self._start_time

    def get_dl_size(self) -> int:
        return self._total_size

    def get_dest(self) -> str | None:
        return self._dest

    def cancel(self) -> None:
        self._cancelled.set()

    def wait(self, raise_exceptions: bool = False) -> bool:
        self._finished.wait()
        if raise_exceptions and self._errors:
            raise self._errors[0]
        return self._success

    def ensure_downloaded(self):
        """Wait to finish state download before proceeding. If download fails, raises an exception."""
        while not self._finished.wait(timeout=5):
            progress = self.get_progress()
            eta = self.get_eta()
            eta_str = f"{eta:.0f}s" if eta >= 0 else "unknown"
            speed_mb = self.get_speed() / 1024**2
            logger.info(
                f"State download progress: {progress * 100.0:.2f}%, speed: {speed_mb:.1f} MB/s, ETA: {eta_str}"
            )
        if self._success:
            logger.info(
                f"State ({self.get_dl_size() // 1024**2} MB) downloaded in {self.get_dl_time():.2f} sec to {self.get_dest()}"
            )
        else:
            logger.error("State download failed:")
            for e in self._errors:
                logger.exception(e)
            raise RuntimeError("State download failed")

    def _dest_path(self) -> str:
        parsed = urlparse(self.url)
        filename = unquote(parsed.path.split("/")[-1])
        dest_dir = Path(f"/tmp/agora_downloads_gpu{self.gpu_id}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        return str(dest_dir / filename)

    def _run(self) -> None:
        try:
            self._dest = self._dest_path()

            head = requests.head(self.url, allow_redirects=True, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT))
            head.raise_for_status()
            self._total_size = int(head.headers.get("Content-Length", 0))
            accepts_ranges = head.headers.get("Accept-Ranges", "none").lower() == "bytes"

            self._start_time = time.time()

            if accepts_ranges and self._total_size > 0 and self.dl_threads > 1:
                self._download_parallel()
            else:
                self._download_single()

            self._end_time = time.time()
            self._success = True
        except Exception as e:
            self._errors.append(e)
            logger.error(f"State download failed: {e}")
        finally:
            self._finished.set()

    def _download_parallel(self) -> None:
        """Download using parallel Range requests, writing directly to the final file at correct offsets."""
        chunk_size = self._total_size // self.dl_threads
        ranges = [
            (i * chunk_size, (i + 1) * chunk_size - 1 if i < self.dl_threads - 1 else self._total_size - 1)
            for i in range(self.dl_threads)
        ]

        # Pre-allocate the full file so all threads can write at their offsets immediately
        with open(self._dest, "wb") as f:
            os.truncate(f.fileno(), self._total_size)

        fd = os.open(self._dest, os.O_WRONLY)
        try:
            with ThreadPoolExecutor(max_workers=self.dl_threads) as pool:
                futures = {pool.submit(self._download_chunk, fd, start, end): (start, end) for start, end in ranges}
                for future in as_completed(futures):
                    if self._cancelled.is_set():
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise RuntimeError("Download cancelled")
                    future.result()  # re-raises chunk exceptions into the main download thread
        finally:
            os.close(fd)

    def _download_chunk(self, fd: int, start: int, end: int) -> None:
        """Download a byte range and write it directly into the file at the correct offset."""
        for attempt in range(1, _CHUNK_RETRIES + 1):
            if self._cancelled.is_set():
                return
            offset = start
            bytes_this_attempt = 0
            try:
                with requests.Session() as session:
                    with session.get(
                        self.url,
                        headers={"Range": f"bytes={start}-{end}"},
                        stream=True,
                        timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
                    ) as response:
                        response.raise_for_status()
                        for data in response.iter_content(_READ_CHUNK_SIZE):
                            if self._cancelled.is_set():
                                return
                            os.pwrite(fd, data, offset)
                            offset += len(data)
                            bytes_this_attempt += len(data)
                            with self._lock:
                                self._bytes_downloaded += len(data)
                                now = time.time()
                                self._speed_samples.append((now, self._bytes_downloaded))
                                while now - self._speed_samples[0][0] > _SPEED_WINDOW:
                                    self._speed_samples.popleft()
                return  # success
            except Exception as e:
                with self._lock:
                    self._bytes_downloaded -= bytes_this_attempt
                    self._speed_samples.clear()
                if attempt == _CHUNK_RETRIES:
                    raise RuntimeError(f"Chunk {start}-{end} failed after {_CHUNK_RETRIES} attempts") from e
                logger.warning(f"Chunk {start}-{end} attempt {attempt} failed: {e}. Retrying...")
                time.sleep(2**attempt)

    def _download_single(self) -> None:
        """Single-threaded fallback for servers that don't support Range requests."""
        with requests.Session() as session:
            with session.get(self.url, stream=True, timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT)) as response:
                response.raise_for_status()
                if self._total_size == 0:
                    self._total_size = int(response.headers.get("Content-Length", 0))
                with open(self._dest, "wb") as f:
                    for data in response.iter_content(_READ_CHUNK_SIZE):
                        if self._cancelled.is_set():
                            raise RuntimeError("Download cancelled")
                        f.write(data)
                        with self._lock:
                            self._bytes_downloaded += len(data)
                            now = time.time()
                            self._speed_samples.append((now, self._bytes_downloaded))
                            while now - self._speed_samples[0][0] > _SPEED_WINDOW:
                                self._speed_samples.popleft()


class StateLoader:
    """Matches the interface of CheckpointSaver, so that it can be used in Server to load state from downloaded checkpoint."""

    def __init__(self, module_backends: dict[str, ModuleBackend], local_path: str):
        assert len(module_backends) == 1
        self.expert = next(iter(module_backends.values()))
        self.local_path = local_path

    def load_experts(self):
        try:
            self.expert.load_state_dict(torch.load(self.local_path, weights_only=True))
            logger.info(f"Loaded checkpoint into {self.expert.name}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint for expert {self.expert.name}: {e}")
            raise Exception(f"Failed to load checkpoint for expert {self.expert.name}") from e
