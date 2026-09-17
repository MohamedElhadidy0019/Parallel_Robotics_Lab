"""Focused regression tests for SteveSimEnv lifecycle and PyBullet connection management.

Covers:
- Real headless PyBullet leak reproduction when initialization raises.
- Disconnection and re-raising on KeyboardInterrupt.
- Preservation of the original exception during failed init.
- Preservation of an independently connected client across failures and closes.
- Verification that close() disconnects the saved owned client ID (not checking -1).
- Mocked constructor ownership isolation.
- Explicit handling of failed connections (negative client_id or connect error).
- Normal construction and idempotent/repeated close() safe against ID reuse.
- Disconnect failure: client ID retained for retry, error exposed, retry succeeds.
- Dual failure during init and cleanup: original exception preserved, cleanup error observable.
- Comprehensive finally block cleanup in all real-client tests.
"""

from unittest.mock import MagicMock
import pytest
import pybullet as p

from sim.env import SteveSimEnv


def test_real_pybullet_leak_on_load_robot_failure(monkeypatch):
    """Reproduced audit probe: Injected RuntimeError in _load_robot must not leak client."""
    created_cids = []
    real_connect = p.connect

    def tracking_connect(*args, **kwargs):
        cid = real_connect(*args, **kwargs)
        created_cids.append(cid)
        return cid

    monkeypatch.setattr(p, "connect", tracking_connect)

    def failing_load_robot(self):
        raise RuntimeError("Audit probe: simulated failure in _load_robot")

    monkeypatch.setattr(SteveSimEnv, "_load_robot", failing_load_robot)

    try:
        with pytest.raises(RuntimeError, match="Audit probe"):
            SteveSimEnv(render=False)

        assert len(created_cids) == 1
        cid = created_cids[0]
        assert not p.isConnected(physicsClientId=cid), f"PyBullet client {cid} was leaked after init failure!"
    finally:
        for cid in created_cids:
            if p.isConnected(physicsClientId=cid):
                p.disconnect(physicsClientId=cid)


def test_keyboard_interrupt_during_init_cleans_up_and_reraises(monkeypatch):
    """KeyboardInterrupt during init must disconnect client and re-raise KeyboardInterrupt."""
    created_cids = []
    real_connect = p.connect

    def tracking_connect(*args, **kwargs):
        cid = real_connect(*args, **kwargs)
        created_cids.append(cid)
        return cid

    monkeypatch.setattr(p, "connect", tracking_connect)

    def interrupting_load_robot(self):
        raise KeyboardInterrupt("Simulated Ctrl+C during robot load")

    monkeypatch.setattr(SteveSimEnv, "_load_robot", interrupting_load_robot)

    try:
        with pytest.raises(KeyboardInterrupt, match=r"Simulated Ctrl\+C"):
            SteveSimEnv(render=False)

        assert len(created_cids) == 1
        cid = created_cids[0]
        assert not p.isConnected(physicsClientId=cid), f"PyBullet client {cid} was leaked on KeyboardInterrupt!"
    finally:
        for cid in created_cids:
            if p.isConnected(physicsClientId=cid):
                p.disconnect(physicsClientId=cid)


class CustomInitError(Exception):
    def __init__(self, code: int, detail: str):
        super().__init__(code, detail)
        self.code = code
        self.detail = detail


def test_original_exception_preserved(monkeypatch):
    """Original exception type, attributes, and arguments must be preserved exactly."""
    created_cids = []
    real_connect = p.connect

    def tracking_connect(*args, **kwargs):
        cid = real_connect(*args, **kwargs)
        created_cids.append(cid)
        return cid

    monkeypatch.setattr(p, "connect", tracking_connect)

    def failing_place_object(self):
        raise CustomInitError(42, "custom object placement failure")

    monkeypatch.setattr(SteveSimEnv, "_place_object", failing_place_object)

    try:
        with pytest.raises(CustomInitError) as exc_info:
            SteveSimEnv(render=False)

        assert exc_info.value.code == 42
        assert exc_info.value.detail == "custom object placement failure"
    finally:
        for cid in created_cids:
            if p.isConnected(physicsClientId=cid):
                p.disconnect(physicsClientId=cid)


def test_preservation_of_independent_client_on_failure(monkeypatch):
    """An independently connected client must not be disconnected when SteveSimEnv init fails."""
    indep_cid = p.connect(p.DIRECT)
    assert p.isConnected(physicsClientId=indep_cid), "Pre-existing client failed to connect"

    created_cids = []
    real_connect = p.connect

    def tracking_connect(*args, **kwargs):
        cid = real_connect(*args, **kwargs)
        created_cids.append(cid)
        return cid

    monkeypatch.setattr(p, "connect", tracking_connect)

    try:
        def failing_load_robot(self):
            raise RuntimeError("Failure in _load_robot while another client exists")

        monkeypatch.setattr(SteveSimEnv, "_load_robot", failing_load_robot)

        with pytest.raises(RuntimeError, match="while another client exists"):
            SteveSimEnv(render=False)

        # Independent client must still be connected
        assert p.isConnected(physicsClientId=indep_cid), "Independent client was disconnected by init cleanup!"
        # Owned client must be disconnected
        for cid in created_cids:
            assert not p.isConnected(physicsClientId=cid), f"Owned client {cid} remained connected!"
    finally:
        for cid in created_cids:
            if p.isConnected(physicsClientId=cid):
                p.disconnect(physicsClientId=cid)
        if p.isConnected(physicsClientId=indep_cid):
            p.disconnect(physicsClientId=indep_cid)


def test_preservation_of_independent_client_on_successful_close():
    """An independently connected client must survive env.close(); verified using saved owned ID."""
    indep_cid = p.connect(p.DIRECT)
    assert p.isConnected(physicsClientId=indep_cid)

    env = None
    owned_cid = -1
    try:
        env = SteveSimEnv(render=False)
        owned_cid = env.client_id
        assert owned_cid >= 0, "Constructor did not set a valid client ID"
        assert owned_cid != indep_cid, "Constructor reused independent client ID"
        assert p.isConnected(physicsClientId=owned_cid), "Owned client should be connected"

        env.close()

        # Check the saved owned client ID directly -- not the reset env.client_id attribute
        assert not p.isConnected(physicsClientId=owned_cid), f"Owned client {owned_cid} remained connected after close!"
        assert p.isConnected(physicsClientId=indep_cid), "Independent client was disconnected by env.close()!"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if owned_cid >= 0 and p.isConnected(physicsClientId=owned_cid):
            p.disconnect(physicsClientId=owned_cid)
        if p.isConnected(physicsClientId=indep_cid):
            p.disconnect(physicsClientId=indep_cid)


def test_constructor_ownership_isolated_with_mock(monkeypatch):
    """Mock test to verify constructor cleanup strictly targets its own client ID."""
    mock_p = MagicMock()
    mock_p.connect.return_value = 77
    mock_p.isConnected.return_value = True
    mock_p.loadURDF.side_effect = RuntimeError("plane load failed")

    monkeypatch.setattr("sim.env.p", mock_p)

    with pytest.raises(RuntimeError, match="plane load failed"):
        SteveSimEnv(render=False)

    mock_p.disconnect.assert_called_once_with(physicsClientId=77)


def test_failed_connection_handling_negative_cid(monkeypatch):
    """When p.connect returns < 0, an explicit ConnectionError is raised and no disconnect occurs."""
    mock_p = MagicMock()
    mock_p.connect.return_value = -1

    monkeypatch.setattr("sim.env.p", mock_p)

    with pytest.raises(ConnectionError, match="Failed to connect to PyBullet"):
        SteveSimEnv(render=False)

    mock_p.disconnect.assert_not_called()


def test_failed_connection_handling_connect_raises(monkeypatch):
    """When p.connect raises an exception, it propagates and no disconnect occurs."""
    mock_p = MagicMock()
    mock_p.connect.side_effect = p.error("Mocked connect failure")

    monkeypatch.setattr("sim.env.p", mock_p)

    with pytest.raises(p.error, match="Mocked connect failure"):
        SteveSimEnv(render=False)

    mock_p.disconnect.assert_not_called()


def test_successful_construction_and_repeated_close():
    """Normal construction remains connected until close(); repeated close() is safe and idempotent."""
    env = None
    owned_cid = -1
    new_cid = -1
    try:
        env = SteveSimEnv(render=False)
        owned_cid = env.client_id
        assert owned_cid >= 0
        assert p.isConnected(physicsClientId=owned_cid)

        # First close
        env.close()
        assert not p.isConnected(physicsClientId=owned_cid)
        assert env.client_id == -1

        # Repeated close calls must not error
        env.close()
        env.close()

        # If a new client reuses the old client ID, calling close() on old env must not touch it
        new_cid = p.connect(p.DIRECT)
        assert p.isConnected(physicsClientId=new_cid)
        env.close()
        assert p.isConnected(physicsClientId=new_cid), "Repeated close() on old env disconnected new client!"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if owned_cid >= 0 and p.isConnected(physicsClientId=owned_cid):
            p.disconnect(physicsClientId=owned_cid)
        if new_cid >= 0 and p.isConnected(physicsClientId=new_cid):
            p.disconnect(physicsClientId=new_cid)


def test_disconnect_raises_once_retains_id_and_retry_succeeds(monkeypatch):
    """If disconnect fails while connected, client ID is retained, error is raised, and retry succeeds."""
    mock_p = MagicMock()
    mock_p.connect.return_value = 55
    mock_p.isConnected.return_value = True

    attempts = {"count": 0}

    def flaky_disconnect(physicsClientId):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("Temporary PyBullet disconnect RPC failure")
        mock_p.isConnected.return_value = False

    mock_p.disconnect.side_effect = flaky_disconnect

    monkeypatch.setattr("sim.env.p", mock_p)
    monkeypatch.setattr(SteveSimEnv, "_load_robot", lambda self: None)
    monkeypatch.setattr(SteveSimEnv, "_place_object", lambda self: None)
    monkeypatch.setattr("sim.env.Table", lambda p_mod, cfg: MagicMock())

    env = SteveSimEnv(render=False)
    assert env.client_id == 55

    # First close: disconnect raises and error is exposed
    with pytest.raises(RuntimeError, match="Temporary PyBullet disconnect RPC failure"):
        env.close()

    # Client ID must still be retained for retry
    assert env.client_id == 55

    # Retry close: disconnect succeeds and client ownership is cleared
    env.close()
    assert env.client_id == -1
    assert attempts["count"] == 2


def test_init_and_cleanup_both_fail_preserves_original_and_observes_cleanup(monkeypatch):
    """When init raises and close also fails, original exception is raised and cleanup failure is observable."""
    mock_p = MagicMock()
    mock_p.connect.return_value = 66
    mock_p.isConnected.return_value = True
    mock_p.loadURDF.side_effect = ValueError("Original init URDF error")
    mock_p.disconnect.side_effect = RuntimeError("Secondary disconnect failure")

    monkeypatch.setattr("sim.env.p", mock_p)

    with pytest.raises(ValueError, match="Original init URDF error") as excinfo:
        SteveSimEnv(render=False)

    err = excinfo.value
    # Original exception is preserved without masking
    assert isinstance(err, ValueError)
    assert str(err) == "Original init URDF error"

    # Cleanup failure is attached and observable on the exception
    cleanup_err = getattr(err, "cleanup_error", getattr(err, "__context__", None))
    assert cleanup_err is not None
    assert isinstance(cleanup_err, RuntimeError)
    assert "Secondary disconnect failure" in str(cleanup_err)


def test_disconnect_failure_followed_by_status_query_failure_retains_id(monkeypatch):
    """When disconnect raises and isConnected query also fails, ownership is retained and failure exposed."""
    mock_p = MagicMock()
    mock_p.connect.return_value = 88
    mock_p.isConnected.return_value = True

    monkeypatch.setattr("sim.env.p", mock_p)
    monkeypatch.setattr(SteveSimEnv, "_load_robot", lambda self: None)
    monkeypatch.setattr(SteveSimEnv, "_place_object", lambda self: None)
    monkeypatch.setattr("sim.env.Table", lambda p_mod, cfg: MagicMock())

    env = SteveSimEnv(render=False)
    assert env.client_id == 88

    # Disconnect raises and isConnected query also raises
    mock_p.disconnect.side_effect = RuntimeError("PyBullet disconnect RPC failure")
    mock_p.isConnected.side_effect = RuntimeError("PyBullet isConnected query failure")

    with pytest.raises(RuntimeError, match="PyBullet disconnect RPC failure"):
        env.close()

    # Unknown status must NOT become successful cleanup: ownership must be retained!
    assert env.client_id == 88


def test_init_failure_with_disconnect_and_status_query_failure_preserves_original(monkeypatch):
    """Original constructor exception must be preserved when cleanup has dual disconnect and query failure."""
    mock_p = MagicMock()
    mock_p.connect.return_value = 89
    mock_p.isConnected.return_value = True
    mock_p.loadURDF.side_effect = KeyError("Missing essential robot URDF link")
    mock_p.disconnect.side_effect = RuntimeError("PyBullet disconnect RPC failure")

    def query_status(physicsClientId=0):
        if mock_p.disconnect.called:
            raise RuntimeError("PyBullet isConnected query failure")
        return True

    mock_p.isConnected.side_effect = query_status

    monkeypatch.setattr("sim.env.p", mock_p)

    with pytest.raises(KeyError, match="Missing essential robot URDF link") as excinfo:
        SteveSimEnv(render=False)

    err = excinfo.value
    assert isinstance(err, KeyError)
    assert "Missing essential robot URDF link" in str(err)

    cleanup_err = getattr(err, "cleanup_error", getattr(err, "__context__", None))
    assert cleanup_err is not None
    assert isinstance(cleanup_err, RuntimeError)
    assert "PyBullet disconnect RPC failure" in str(cleanup_err)
