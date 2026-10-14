import errno
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from neos.workflow import shared_state_io as state


def test_stale_read_reopens_and_reads_current_version(tmp_path):
    path = tmp_path / 'serving.json'
    with patch.object(Path, 'read_text', side_effect=[
        OSError(errno.ESTALE, 'Stale file handle'),
        json.dumps({'status': 'ready', 'version': 1}),
    ]) as read, patch.object(state.time, 'sleep'):
        assert state.read_json(path)['version'] == 1
        assert read.call_count == 2


def test_persistent_stale_is_not_reported_ready(tmp_path, monkeypatch):
    monkeypatch.setenv('NEOS_STATE_IO_ATTEMPTS', '3')
    with patch.object(Path, 'read_text', side_effect=OSError(errno.ESTALE, 'stale')) as read, \
         patch.object(state.time, 'sleep'), pytest.raises(OSError) as result:
        state.read_json(tmp_path / 'serving.json', {'status': 'ready'})
    assert result.value.errno == errno.ESTALE
    assert read.call_count == 3


def test_missing_state_and_permission_error(tmp_path):
    assert state.read_json(tmp_path / 'missing', {}) == {}
    with patch.object(Path, 'read_text', side_effect=PermissionError(errno.EACCES, 'denied')) as read, \
         pytest.raises(PermissionError):
        state.read_json(tmp_path / 'denied')
    assert read.call_count == 1


def test_replacement_retry_keeps_previous_json_until_commit(tmp_path):
    path = tmp_path / 'serving.json'
    state.atomic_json(path, {'version': 0})
    replace = state.os.replace
    calls = []

    def fail_once(source, target):
        calls.append(str(source))
        if len(calls) == 1:
            assert state.read_json(target) == {'version': 0}
            raise OSError(errno.ESTALE, 'stale rename')
        replace(source, target)

    with patch.object(state.os, 'replace', side_effect=fail_once), patch.object(state.time, 'sleep'):
        state.atomic_json(path, {'version': 1})
    assert len(set(calls)) == 2
    assert state.read_json(path) == {'version': 1}
    assert list(tmp_path.iterdir()) == [path]


def test_failed_state_is_preserved_after_retry(tmp_path):
    with patch.object(Path, 'read_text', side_effect=[
        json.JSONDecodeError('partial', '{', 1), '{"status":"failed","error":"SFT failed"}',
    ]), patch.object(state.time, 'sleep'):
        assert state.read_json(tmp_path / 'serving.json')['status'] == 'failed'
