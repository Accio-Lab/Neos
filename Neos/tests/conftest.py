
import socket
import pytest


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    original=socket.socket.connect
    def connect(sock,address):
        if isinstance(address,tuple) and address[0] not in {'localhost','127.0.0.1','::1'}:
            raise RuntimeError('External network access is disabled in Neos unit tests')
        return original(sock,address)
    monkeypatch.setattr(socket.socket,'connect',connect)
