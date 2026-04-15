"""Shared network utilities: length-prefixed pickle protocol."""
import pickle
import socket
import struct
from typing import Any


def send_msg(sock: socket.socket, data: Any) -> None:
    """Send a length-prefixed pickled message."""
    payload = pickle.dumps(data)
    sock.sendall(struct.pack('>I', len(payload)) + payload)


def recv_msg(sock: socket.socket) -> Any:
    """Receive a length-prefixed pickled message. Returns None if connection closed."""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    msg_len = struct.unpack('>I', header)[0]
    payload = _recv_exact(sock, msg_len)
    if payload is None:
        return None
    return pickle.loads(payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
