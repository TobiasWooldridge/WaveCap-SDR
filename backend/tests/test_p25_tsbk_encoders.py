from wavecapsdr.decoders.p25_tsbk import TSBKOpcode, TSBKParser
from wavecapsdr.decoders.p25_tsbk_encoders import (
    encode_group_affiliation_response,
    encode_status_update,
    encode_unit_deregistration_ack,
    encode_unit_deregistration_request,
    encode_unit_registration_request,
    encode_unit_registration_response,
    encode_unit_service_request,
)
from wavecapsdr.decoders.tsbk_utils import payload_to_bits, read_field
import pytest


class TestTsbkEncoders:
    def setup_method(self) -> None:
        self.parser = TSBKParser()

    @staticmethod
    def _decode_unit_registration_request(payload: bytes) -> dict[str, int | bool]:
        bits = payload_to_bits(payload)
        emergency = bool(read_field(bits, 0, 1))
        capability = read_field(bits, 1, 7)
        wacn = read_field(bits, 8, 20)
        system_id = read_field(bits, 28, 12)
        source_id = read_field(bits, 40, 24)
        return {
            "emergency": emergency,
            "capability": capability,
            "wacn": wacn,
            "system_id": system_id,
            "source_id": source_id,
        }

    @staticmethod
    def _decode_unit_deregistration_request(payload: bytes) -> dict[str, int]:
        bits = payload_to_bits(payload)
        wacn = read_field(bits, 8, 20)
        system_id = read_field(bits, 28, 12)
        source_id = read_field(bits, 40, 24)
        return {
            "wacn": wacn,
            "system_id": system_id,
            "source_id": source_id,
        }

    def test_status_update_round_trip(self) -> None:
        payload = encode_status_update(0x12, 0x34, target_id=0xABCDE, source_id=0x123456)
        parsed = self.parser.parse(TSBKOpcode.STATUS_UPDT, 0, payload)

        assert parsed["type"] == "STATUS_UPDATE"
        assert parsed["unit_status"] == 0x12
        assert parsed["user_status"] == 0x34
        assert parsed["target_id"] == 0xABCDE
        assert parsed["source_id"] == 0x123456

    def test_group_affiliation_response_round_trip(self) -> None:
        payload = encode_group_affiliation_response(
            response_code=1,
            announcement_group=0x1111,
            group_address=0x2222,
            target_id=0xABCDE,
            global_affiliation=True,
        )
        parsed = self.parser.parse(TSBKOpcode.GRP_AFF_RSP, 0, payload)

        assert parsed["type"] == "GROUP_AFFILIATION_RESPONSE"
        assert parsed["response"] == 1
        assert parsed["announcement_group"] == 0x1111
        assert parsed["tgid"] == 0x2222
        assert parsed["target_id"] == 0xABCDE
        assert parsed["global"] is True

    def test_unit_registration_round_trip(self) -> None:
        payload = encode_unit_registration_response(
            response_code=0,
            system_id=0x321,
            source_id=0x123456,
            registered_address=0x0FEDCB,
        )
        parsed = self.parser.parse(TSBKOpcode.UNIT_REG_RSP, 0, payload)

        assert parsed["type"] == "UNIT_REGISTRATION_RESPONSE"
        assert parsed["system_id"] == 0x321
        assert parsed["source_id"] == 0x123456
        assert parsed["registered_address"] == 0x0FEDCB
        assert parsed["response"] == 0
        assert parsed["success"] is True

    def test_unit_dereg_ack_round_trip(self) -> None:
        payload = encode_unit_deregistration_ack(
            wacn=0xABCDE,
            system_id=0x321,
            target_id=0x123456,
        )
        parsed = self.parser.parse(TSBKOpcode.UNIT_DEREG_ACK, 0, payload)

        assert parsed["type"] == "UNIT_DEREGISTRATION_ACK"
        assert parsed["wacn"] == 0xABCDE
        assert parsed["system_id"] == 0x321
        assert parsed["target_id"] == 0x123456

    def test_unit_service_request_round_trip(self) -> None:
        payload = encode_unit_service_request(
            service_options=0xA5,
            target_id=0x00F001,
            source_id=0x001122,
        )
        parsed = self.parser.parse(TSBKOpcode.UU_ANS_REQ, 0, payload)

        assert parsed["type"] == "UNIT_SERVICE_REQUEST"
        assert parsed["service_options"] == 0xA5
        assert parsed["target_id"] == 0x00F001
        assert parsed["source_id"] == 0x001122

    def test_unit_registration_request_round_trip(self) -> None:
        payload = encode_unit_registration_request(
            wacn=0xABCDE,
            system_id=0x321,
            source_id=0x00BEEF,
            capability=0x45,
            emergency=True,
        )
        parsed = self._decode_unit_registration_request(payload)

        assert parsed["emergency"] is True
        assert parsed["capability"] == 0x45
        assert parsed["wacn"] == 0xABCDE
        assert parsed["system_id"] == 0x321
        assert parsed["source_id"] == 0x00BEEF

    def test_unit_deregistration_request_round_trip(self) -> None:
        payload = encode_unit_deregistration_request(
            wacn=0xABCDE,
            system_id=0x321,
            source_id=0x00ABCD,
        )
        parsed = self._decode_unit_deregistration_request(payload)

        assert parsed["wacn"] == 0xABCDE
        assert parsed["system_id"] == 0x321
        assert parsed["source_id"] == 0x00ABCD

    @pytest.mark.parametrize(
        "encoder,args",
        [
            (encode_status_update, {"unit_status": 0x1FF, "user_status": 0, "target_id": 1, "source_id": 1}),
            (encode_unit_registration_request, {"wacn": 0x200000, "system_id": 1, "source_id": 1}),
            (encode_unit_deregistration_request, {"wacn": 1, "system_id": 0x2000, "source_id": 1}),
            (encode_unit_registration_response, {"response_code": 5, "system_id": 1, "source_id": 1}),
        ],
    )
    def test_encoder_validates_ranges(self, encoder, args) -> None:
        with pytest.raises(ValueError):
            encoder(**args)
