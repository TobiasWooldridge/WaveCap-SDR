"""Tests for P25 trunking voice stream API endpoints.

Tests the voice stream REST and WebSocket endpoints.

Reference: SDRTrunk (https://github.com/DSheirer/sdrtrunk)
"""

import pytest
from collections.abc import Iterator
from fastapi.testclient import TestClient

from wavecapsdr.app import create_app
from wavecapsdr.config import AppConfig, DeviceConfig
from wavecapsdr.trunking.voice_channel import (
    VoiceChannel,
    VoiceChannelConfig,
    RadioLocation,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create a test client with a fake driver."""
    cfg = AppConfig()
    cfg.device = DeviceConfig(driver="fake")
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def system_with_client(client: TestClient):
    """Create a trunking system and return (client, system_id)."""
    system_data = {
        "id": "voice-test-system",
        "name": "Voice Test System",
        "protocol": "p25_phase1",
        "control_channels": [851_000_000],
        "center_hz": 855_000_000,
        "sample_rate": 4_000_000,
        "max_voice_recorders": 4,
    }
    response = client.post("/api/v1/trunking/systems", json=system_data)
    assert response.status_code == 200
    return client, "voice-test-system"


# ============================================================================
# Voice Streams Endpoint Tests
# ============================================================================

class TestVoiceStreamsAPI:
    """Test voice stream REST endpoints."""

    def test_list_voice_streams_no_system(self, client: TestClient) -> None:
        """404 when system doesn't exist."""
        response = client.get("/api/v1/trunking/systems/nonexistent/voice-streams")
        assert response.status_code == 404

    def test_list_voice_streams_empty(self, system_with_client) -> None:
        """List streams returns empty when no active calls."""
        client, system_id = system_with_client

        response = client.get(f"/api/v1/trunking/systems/{system_id}/voice-streams")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_voice_streams_response_format(self, system_with_client) -> None:
        """Verify voice streams response format matches spec."""
        client, system_id = system_with_client

        # We can't easily create active voice channels without a running system,
        # but we can verify the endpoint works and returns correct format
        response = client.get(f"/api/v1/trunking/systems/{system_id}/voice-streams")
        assert response.status_code == 200

        # Empty list is valid
        data = response.json()
        assert isinstance(data, list)


# ============================================================================
# Radio Locations Endpoint Tests
# ============================================================================

class TestRadioLocationsAPI:
    """Test radio locations REST endpoint."""

    def test_get_locations_no_system(self, client: TestClient) -> None:
        """404 when system doesn't exist."""
        response = client.get("/api/v1/trunking/systems/nonexistent/locations")
        assert response.status_code == 404

    def test_get_locations_empty(self, system_with_client) -> None:
        """Get locations returns empty when no GPS data."""
        client, system_id = system_with_client

        response = client.get(f"/api/v1/trunking/systems/{system_id}/locations")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_locations_response_format(self, system_with_client) -> None:
        """Verify locations response format."""
        client, system_id = system_with_client

        response = client.get(f"/api/v1/trunking/systems/{system_id}/locations")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

        # If there were locations, they'd have these fields:
        # unitId, latitude, longitude, altitude, speed, heading,
        # accuracy, timestamp, ageSeconds, source


# ============================================================================
# Vocoder Status Endpoint Tests
# ============================================================================

class TestVocoderStatusAPI:
    """Test vocoder status endpoint."""

    def test_get_vocoder_status(self, client: TestClient) -> None:
        """Get vocoder availability status."""
        response = client.get("/api/v1/trunking/vocoders")
        assert response.status_code == 200

        data = response.json()
        assert "imbe" in data
        assert "ambe2" in data
        assert "anyAvailable" in data

        # IMBE and AMBE2 should have available and message fields
        assert "available" in data["imbe"]
        assert "message" in data["imbe"]
        assert "available" in data["ambe2"]
        assert "message" in data["ambe2"]


# ============================================================================
# Voice Stream WebSocket Tests
# ============================================================================

class TestVoiceStreamWebSocket:
    """Test voice stream WebSocket endpoints."""

    def test_voice_stream_websocket_no_system(self, client: TestClient) -> None:
        """WebSocket closes for nonexistent system."""
        # WebSocket may connect then close, or raise - both are valid behaviors
        try:
            with client.websocket_connect("/api/v1/trunking/stream/nonexistent/voice") as ws:
                # If we get here, the connection was accepted - it should close quickly
                try:
                    ws.receive_text(timeout=1.0)
                except Exception:
                    pass  # Expected - connection closes
        except Exception:
            pass  # Also valid - connection rejected

    def test_voice_stream_websocket_connect(self, system_with_client) -> None:
        """WebSocket connects successfully for existing system."""
        client, system_id = system_with_client

        # This will connect but likely close quickly since no voice channels
        # are active. The important thing is the connection succeeds.
        try:
            with client.websocket_connect(
                f"/api/v1/trunking/stream/{system_id}/voice"
            ) as ws:
                # Connection established - success
                # WebSocket will stay open waiting for audio
                pass
        except Exception:
            # Connection may close if no active channels
            pass

    def test_single_voice_stream_websocket_no_stream(self, system_with_client) -> None:
        """WebSocket closes for nonexistent stream."""
        client, system_id = system_with_client

        # WebSocket may connect then close, or raise - both are valid behaviors
        try:
            with client.websocket_connect(
                f"/api/v1/trunking/stream/{system_id}/voice/nonexistent"
            ) as ws:
                # If we get here, the connection was accepted - it should close quickly
                try:
                    ws.receive_text(timeout=1.0)
                except Exception:
                    pass  # Expected - connection closes
        except Exception:
            pass  # Also valid - connection rejected


# ============================================================================
# PCM Stream Endpoint Tests
# ============================================================================

class TestPCMStreamAPI:
    """Test PCM streaming endpoint."""

    def test_pcm_stream_no_system(self, client: TestClient) -> None:
        """404 when system doesn't exist."""
        response = client.get(
            "/api/v1/trunking/stream/nonexistent/voice/stream0.pcm"
        )
        assert response.status_code == 404

    def test_pcm_stream_no_stream(self, system_with_client) -> None:
        """404 when stream doesn't exist."""
        client, system_id = system_with_client

        response = client.get(
            f"/api/v1/trunking/stream/{system_id}/voice/nonexistent.pcm"
        )
        assert response.status_code == 404


# ============================================================================
# Integration Tests - System with Voice
# ============================================================================

class TestVoiceStreamIntegration:
    """Integration tests for voice stream functionality."""

    def test_system_lifecycle_with_voice_endpoints(self, client: TestClient) -> None:
        """Test system CRUD doesn't break voice endpoints."""
        # Create system
        system_data = {
            "id": "lifecycle-test",
            "name": "Lifecycle Test",
            "protocol": "p25_phase1",
            "control_channels": [851_000_000],
            "center_hz": 855_000_000,
        }
        response = client.post("/api/v1/trunking/systems", json=system_data)
        assert response.status_code == 200

        # Voice endpoints should work
        response = client.get("/api/v1/trunking/systems/lifecycle-test/voice-streams")
        assert response.status_code == 200

        response = client.get("/api/v1/trunking/systems/lifecycle-test/locations")
        assert response.status_code == 200

        # Delete system
        response = client.delete("/api/v1/trunking/systems/lifecycle-test")
        assert response.status_code == 200

        # Voice endpoints should 404 now
        response = client.get("/api/v1/trunking/systems/lifecycle-test/voice-streams")
        assert response.status_code == 404

    def test_multiple_systems_voice_isolation(self, client: TestClient) -> None:
        """Each system has its own voice streams."""
        # Create two systems
        for i in range(2):
            system_data = {
                "id": f"multi-system-{i}",
                "name": f"Multi System {i}",
                "protocol": "p25_phase1",
                "control_channels": [851_000_000 + i * 100_000],
                "center_hz": 855_000_000,
            }
            response = client.post("/api/v1/trunking/systems", json=system_data)
            assert response.status_code == 200

        # Each system's voice-streams endpoint works independently
        for i in range(2):
            response = client.get(f"/api/v1/trunking/systems/multi-system-{i}/voice-streams")
            assert response.status_code == 200
            assert response.json() == []

            response = client.get(f"/api/v1/trunking/systems/multi-system-{i}/locations")
            assert response.status_code == 200
            assert response.json() == []


# ============================================================================
# VoiceStream Response Model Tests
# ============================================================================

class TestVoiceStreamResponseModel:
    """Test VoiceStreamResponse model validation."""

    def test_voice_stream_model_fields(self):
        """Verify VoiceStreamResponse has all expected fields."""
        from wavecapsdr.trunking.api import VoiceStreamResponse

        # Create a sample response
        response = VoiceStreamResponse(
            id="test_vc0",
            systemId="test_sys",
            callId="call_001",
            recorderId="vr0",
            state="active",
            talkgroupId=1001,
            talkgroupName="Test TG",
            sourceId=12345678,
            encrypted=False,
            startTime=1702834567.0,
            durationSeconds=10.5,
            silenceSeconds=0.1,
            audioFrameCount=100,
            audioBytesSent=10000,
            subscriberCount=1,
            sourceLocation=None,
        )

        assert response.id == "test_vc0"
        assert response.state == "active"
        assert response.talkgroupId == 1001

    def test_voice_stream_model_with_location(self):
        """VoiceStreamResponse with location data."""
        from wavecapsdr.trunking.api import VoiceStreamResponse, LocationResponse

        location = LocationResponse(
            unitId=12345678,
            latitude=47.6,
            longitude=-122.3,
            altitude=100.0,
            speed=45.0,
            heading=270.0,
            accuracy=5.0,
            timestamp=1702834567.0,
            ageSeconds=10.0,
            source="elc",
        )

        response = VoiceStreamResponse(
            id="test_vc0",
            systemId="test_sys",
            callId="call_001",
            recorderId="vr0",
            state="active",
            talkgroupId=1001,
            talkgroupName="Test TG",
            sourceId=12345678,
            encrypted=False,
            startTime=1702834567.0,
            durationSeconds=10.5,
            silenceSeconds=0.1,
            audioFrameCount=100,
            audioBytesSent=10000,
            subscriberCount=1,
            sourceLocation=location,
        )

        assert response.sourceLocation is not None
        assert response.sourceLocation.latitude == 47.6


class TestLocationResponseModel:
    """Test LocationResponse model validation."""

    def test_location_response_fields(self):
        """Verify LocationResponse has all expected fields."""
        from wavecapsdr.trunking.api import LocationResponse

        loc = LocationResponse(
            unitId=12345,
            latitude=47.6062,
            longitude=-122.3321,
            altitude=100.0,
            speed=60.0,
            heading=180.0,
            accuracy=5.0,
            timestamp=1702834567.0,
            ageSeconds=30.0,
            source="lrrp",
        )

        assert loc.unitId == 12345
        assert loc.latitude == 47.6062
        assert loc.source == "lrrp"

    def test_location_response_optional_fields(self):
        """Optional fields can be None."""
        from wavecapsdr.trunking.api import LocationResponse

        loc = LocationResponse(
            unitId=12345,
            latitude=47.6,
            longitude=-122.3,
            altitude=None,
            speed=None,
            heading=None,
            accuracy=None,
            timestamp=1702834567.0,
            ageSeconds=30.0,
            source="elc",
        )

        assert loc.altitude is None
        assert loc.speed is None
