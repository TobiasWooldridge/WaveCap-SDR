"""Tests for RadioReference talkgroup parsing and import wiring."""
from __future__ import annotations

from pathlib import Path

from wavecapsdr.config import RadioReferenceConfig
from wavecapsdr.radioreference import (
    RadioReferenceTalkgroup,
    parse_talkgroups_response,
)
from wavecapsdr.trunking.config import (
    RadioReferenceTalkgroupsConfig,
    load_talkgroups_radioreference,
)


SOAP_RESPONSE = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"
    xmlns:tns="http://api.radioreference.com/soap2"
    xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <soap:Body>
    <tns:getTrsTalkgroupsResponse>
      <return SOAP-ENC:arrayType="tns:Talkgroup[2]">
        <item xsi:type="tns:Talkgroup">
          <tgDec>100</tgDec>
          <tgDescr>Admin</tgDescr>
          <tgAlpha>ADMIN</tgAlpha>
          <tgSubfleet>Police</tgSubfleet>
          <tgMode>D</tgMode>
          <tags SOAP-ENC:arrayType="tns:tag[1]">
            <item xsi:type="tns:tag">
              <tagId>1</tagId>
              <tagDescr>Law Dispatch</tagDescr>
            </item>
          </tags>
        </item>
        <item xsi:type="tns:Talkgroup">
          <tgDec>200</tgDec>
          <tgDescr>Fire Dispatch</tgDescr>
          <tgAlpha>FIRE</tgAlpha>
          <tgSubfleet></tgSubfleet>
          <tags SOAP-ENC:arrayType="tns:tag[1]">
            <item xsi:type="tns:tag">
              <tagId>2</tagId>
              <tagDescr>Fire</tagDescr>
            </item>
          </tags>
        </item>
      </return>
    </tns:getTrsTalkgroupsResponse>
  </soap:Body>
</soap:Envelope>
"""


def test_parse_talkgroups_response() -> None:
    talkgroups = parse_talkgroups_response(SOAP_RESPONSE)

    assert talkgroups[100].name == "Admin"
    assert talkgroups[100].alpha_tag == "ADMIN"
    assert talkgroups[100].category == "Police"

    assert talkgroups[200].name == "Fire Dispatch"
    assert talkgroups[200].alpha_tag == "FIRE"
    assert talkgroups[200].category == "Fire"


def test_load_talkgroups_radioreference_writes_cache(tmp_path, monkeypatch) -> None:
    def fake_fetch(_config, _req):
        return {
            300: RadioReferenceTalkgroup(
                tgid=300,
                name="Ops",
                alpha_tag="OPS",
                category="Ops",
                mode="D",
            )
        }

    monkeypatch.setattr(
        "wavecapsdr.trunking.config.fetch_talkgroups",
        fake_fetch,
    )

    rr_config = RadioReferenceConfig(
        enabled=True,
        username="user",
        password="pass",
        app_key="key",
    )

    settings = RadioReferenceTalkgroupsConfig(
        system_id=1234,
        cache_file="talkgroups_rr.yaml",
        refresh=True,
    )

    result = load_talkgroups_radioreference(settings, rr_config, config_dir=str(tmp_path))

    assert 300 in result
    cache_path = Path(tmp_path) / "talkgroups_rr.yaml"
    assert cache_path.exists()
