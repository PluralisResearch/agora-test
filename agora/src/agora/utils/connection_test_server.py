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

import socket
import threading
import time

from agora_server.hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class TestServerError(Exception):
    pass


class TestServer:
    def __init__(self, host="0.0.0.0", port=49200):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.thread = None
        self.auth_received = threading.Event()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        """Start the server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
            self.server_socket.settimeout(1)
            self.running = True
        except OSError as e:
            logger.error(f"Failed to bind to {self.host}:{self.port}: {e}")
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            raise TestServerError(f"Failed to start server on {self.host}:{self.port}: {e}") from None

        def server_loop():
            logger.info(f"Test server listening on {self.host}:{self.port}")
            while self.running and not self.auth_received.is_set():
                try:
                    client_socket, addr = self.server_socket.accept()
                    logger.info("Test server connection received")

                    # Handle each client in a separate thread to avoid blocking
                    client_thread = threading.Thread(target=self._handle_client, args=(client_socket, addr))
                    client_thread.daemon = True
                    client_thread.start()

                except TimeoutError:
                    continue
                except OSError:
                    break

        self.thread = threading.Thread(target=server_loop)
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)  # Let server start

    def _handle_client(self, client_socket, addr):
        """Handle individual client connection"""
        try:
            client_socket.settimeout(2)
            data = client_socket.recv(1024)
            if data:
                message = data.decode("utf-8", errors="ignore")

                if self._is_auth_message(message):
                    self.auth_received.set()
                    logger.info("Auth message received!")
                else:
                    logger.info("Non-auth message received")
            else:
                logger.info("No data received")
        except TimeoutError:
            logger.info("Timeout waiting for data")
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
        finally:
            client_socket.close()

    def _is_auth_message(self, message):
        """Check if the message is an auth message"""
        return "auth" in message.lower()

    def close(self):
        """Close the server"""
        if self.running:
            self.running = False
            if self.server_socket:
                self.server_socket.close()
            if self.thread:
                self.thread.join(timeout=1)
            logger.info("Test server closed")

    def wait_for_message(self, timeout=10):
        """Wait for the auth message to be received. Returns True if received, False if timeout"""
        return self.auth_received.wait(timeout)
