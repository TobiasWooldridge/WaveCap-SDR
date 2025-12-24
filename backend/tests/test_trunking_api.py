"""Tests for P25 trunking REST API endpoints.

Tests the trunking API endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from wavecapsdr.app import create_app
from wavecapsdr.config import AppConfig, DeviceConfig


@pytest.fixture
def client() -> TestClient:
    """Create a test client with a fake driver."""
    cfg = AppConfig()
    cfg.device = DeviceConfig(driver="fake")
    app = create_app(cfg)
    return TestClient(app)


class TestTrunkingSystemsAPI:
    """Test trunking systems CRUD endpoints."""

    def test_list_systems_empty(self, client: TestClient) -> None:
        """Test listing systems when none exist."""
        response = client.get("/api/v1/trunking/systems")
        assert response.status_code == 200
        assert response.json() == []

    def test_create_system(self, client: TestClient) -> None:
        """Test creating a new trunking system."""
        system_data = {
            "id": "test-system",
            "name": "Test System",
            "protocol": "p25_phase1",
            "control_channels": [851_000_000, 851_100_000],
            "center_hz": 855_000_000,
            "sample_rate": 4_000_000,
            "max_voice_recorders": 4,
            "auto_start": False,
        }
        response = client.post("/api/v1/trunking/systems", json=system_data)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-system"
        assert data["name"] == "Test System"
        assert data["protocol"] == "p25_phase1"
        assert data["state"] == "stopped"

    def test_create_system_with_talkgroups(self, client: TestClient) -> None:
        """Test creating a system with initial talkgroups."""
        system_data = {
            "id": "test-system-tg",
            "name": "Test System with TGs",
            "protocol": "p25_phase2",
            "control_channels": [851_000_000],
            "center_hz": 855_000_000,
            "sample_rate": 8_000_000,
            "talkgroups": {
                "1217": {
                    "tgid": 1217,
                    "name": "Kirkland PD",
                    "category": "Police",
                    "priority": 1,
                    "record": True,
                    "monitor": True,
                }
            },
        }
        response = client.post("/api/v1/trunking/systems", json=system_data)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-system-tg"
        assert data["protocol"] == "p25_phase2"

    def test_create_system_invalid_protocol(self, client: TestClient) -> None:
        """Test creating a system with invalid protocol."""
        system_data = {
            "id": "bad-system",
            "name": "Bad System",
            "protocol": "invalid_protocol",
            "control_channels": [851_000_000],
            "center_hz": 855_000_000,
        }
        response = client.post("/api/v1/trunking/systems", json=system_data)
        assert response.status_code == 400
        assert "Invalid protocol" in response.json()["detail"]

    def test_get_system(self, client: TestClient) -> None:
        """Test getting a specific system."""
        # Create system first
        system_data = {
            "id": "get-test",
            "name": "Get Test System",
            "protocol": "p25_phase1",
            "control_channels": [851_000_000],
            "center_hz": 855_000_000,
        }
        client.post("/api/v1/trunking/systems", json=system_data)

        # Get the system
        response = client.get("/api/v1/trunking/systems/get-test")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "get-test"
        assert data["name"] == "Get Test System"

    def test_get_system_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent system."""
        response = client.get("/api/v1/trunking/systems/nonexistent")
        assert response.status_code == 404

    def test_delete_system(self, client: TestClient) -> None:
        """Test deleting a system."""
        # Create system first
        system_data = {
            "id": "delete-test",
            "name": "Delete Test System",
            "protocol": "p25_phase1",
            "control_channels": [851_000_000],
            "center_hz": 855_000_000,
        }
        client.post("/api/v1/trunking/systems", json=system_data)

        # Delete the system
        response = client.delete("/api/v1/trunking/systems/delete-test")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # Verify it's gone
        response = client.get("/api/v1/trunking/systems/delete-test")
        assert response.status_code == 404

    def test_delete_system_not_found(self, client: TestClient) -> None:
        """Test deleting a non-existent system."""
        response = client.delete("/api/v1/trunking/systems/nonexistent")
        assert response.status_code == 404


class TestTrunkingSystemLifecycle:
    """Test system start/stop lifecycle."""

    def test_start_system(self, client: TestClient) -> None:
        """Test starting a system."""
        # Create system first
        system_data = {
            "id": "start-test",
            "name": "Start Test System",
            "protocol": "p25_phase1",
            "control_channels": [851_000_000],
            "center_hz": 855_000_000,
        }
        client.post("/api/v1/trunking/systems", json=system_data)

        # Start the system
        response = client.post("/api/v1/trunking/systems/start-test/start")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # Verify state changed
        response = client.get("/api/v1/trunking/systems/start-test")
        assert response.status_code == 200
        # State should be "searching" or "syncing" after start
        assert response.json()["state"] in ["starting", "searching", "syncing", "running"]

    def test_stop_system(self, client: TestClient) -> None:
        """Test stopping a system."""
        # Create and start system
        system_data = {
            "id": "stop-test",
            "name": "Stop Test System",
            "protocol": "p25_phase1",
            "control_channels": [851_000_000],
            "center_hz": 855_000_000,
        }
        client.post("/api/v1/trunking/systems", json=system_data)
        client.post("/api/v1/trunking/systems/stop-test/start")

        # Stop the system
        response = client.post("/api/v1/trunking/systems/stop-test/stop")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # Verify state changed
        response = client.get("/api/v1/trunking/systems/stop-test")
        assert response.status_code == 200
        assert response.json()["state"] == "stopped"

    def test_start_system_not_found(self, client: TestClient) -> None:
        """Test starting a non-existent system."""
        response = client.post("/api/v1/trunking/systems/nonexistent/start")
        assert response.status_code == 404

    def test_stop_system_not_found(self, client: TestClient) -> None:
        """Test stopping a non-existent system."""
        response = client.post("/api/v1/trunking/systems/nonexistent/stop")
        assert response.status_code == 404


class TestTrunkingTalkgroupsAPI:
    """Test talkgroup management endpoints."""

    @pytest.fixture
    def system_with_talkgroups(self, client: TestClient) -> str:
        """Create a system with talkgroups for testing."""
        system_data = {
            "id": "tg-test",
            "name": "Talkgroup Test System",
            "protocol": "p25_phase1",
            "control_channels": [851_000_000],
            "center_hz": 855_000_000,
            "talkgroups": {
                "100": {
                    "tgid": 100,
                    "name": "Admin",
                    "category": "Admin",
                    "priority": 5,
                }
            },
        }
        client.post("/api/v1/trunking/systems", json=system_data)
        return "tg-test"

    def test_get_talkgroups(self, client: TestClient, system_with_talkgroups: str) -> None:
        """Test getting talkgroups for a system."""
        response = client.get(f"/api/v1/trunking/systems/{system_with_talkgroups}/talkgroups")
        assert response.status_code == 200
        talkgroups = response.json()
        assert len(talkgroups) == 1
        assert talkgroups[0]["tgid"] == 100
        assert talkgroups[0]["name"] == "Admin"

    def test_add_talkgroups(self, client: TestClient, system_with_talkgroups: str) -> None:
        """Test adding talkgroups to a system."""
        new_talkgroups = [
            {
                "tgid": 200,
                "name": "Fire Dispatch",
                "category": "Fire",
                "priority": 1,
                "record": True,
                "monitor": True,
            },
            {
                "tgid": 300,
                "name": "EMS",
                "category": "Medical",
                "priority": 2,
                "record": True,
                "monitor": True,
            },
        ]
        response = client.post(
            f"/api/v1/trunking/systems/{system_with_talkgroups}/talkgroups",
            json=new_talkgroups,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["added"] == 2
        assert data["updated"] == 0
        assert data["total"] == 3  # 1 original + 2 new

    def test_update_existing_talkgroup(self, client: TestClient, system_with_talkgroups: str) -> None:
        """Test updating an existing talkgroup."""
        update_talkgroup = [
            {
                "tgid": 100,
                "name": "Admin Updated",
                "category": "Admin",
                "priority": 10,
                "record": False,
                "monitor": True,
            }
        ]
        response = client.post(
            f"/api/v1/trunking/systems/{system_with_talkgroups}/talkgroups",
            json=update_talkgroup,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["added"] == 0
        assert data["updated"] == 1

    def test_get_talkgroups_system_not_found(self, client: TestClient) -> None:
        """Test getting talkgroups for non-existent system."""
        response = client.get("/api/v1/trunking/systems/nonexistent/talkgroups")
        assert response.status_code == 404


class TestTrunkingCallsAPI:
    """Test active calls endpoints."""

    def test_get_all_active_calls_empty(self, client: TestClient) -> None:
        """Test getting all active calls when none exist."""
        response = client.get("/api/v1/trunking/calls")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_system_active_calls_empty(self, client: TestClient) -> None:
        """Test getting system active calls when none exist."""
        # Create a system first
        system_data = {
            "id": "calls-test",
            "name": "Calls Test System",
            "protocol": "p25_phase1",
            "control_channels": [851_000_000],
            "center_hz": 855_000_000,
        }
        client.post("/api/v1/trunking/systems", json=system_data)

        response = client.get("/api/v1/trunking/systems/calls-test/calls/active")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_system_active_calls_not_found(self, client: TestClient) -> None:
        """Test getting active calls for non-existent system."""
        response = client.get("/api/v1/trunking/systems/nonexistent/calls/active")
        assert response.status_code == 404


class TestVocoderStatusAPI:
    """Test vocoder status endpoint."""

    def test_get_vocoder_status(self, client: TestClient) -> None:
        """Test getting vocoder availability status."""
        response = client.get("/api/v1/trunking/vocoders")
        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "imbe" in data
        assert "ambe2" in data
        assert "anyAvailable" in data

        # Verify imbe structure
        assert "available" in data["imbe"]
        assert "message" in data["imbe"]

        # Verify ambe2 structure
        assert "available" in data["ambe2"]
        assert "message" in data["ambe2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
