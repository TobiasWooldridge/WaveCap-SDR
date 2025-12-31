from __future__ import annotations

import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable
from xml.etree import ElementTree

from .config import RadioReferenceConfig

logger = logging.getLogger(__name__)


class RadioReferenceError(RuntimeError):
    pass


@dataclass
class RadioReferenceTalkgroup:
    tgid: int
    name: str
    alpha_tag: str = ""
    category: str = ""
    mode: str = ""


@dataclass
class RadioReferenceTalkgroupRequest:
    system_id: int
    category_id: int | None = None
    tag_id: int | None = None
    tgid: int | None = None


def _soap_text(val: str) -> str:
    return (
        val.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&apos;")
    )


def _build_auth_block(config: RadioReferenceConfig) -> str:
    if not (config.username and config.password and config.app_key):
        raise RadioReferenceError("RadioReference credentials are missing")
    return (
        "<authInfo>"
        f"<username>{_soap_text(config.username)}</username>"
        f"<password>{_soap_text(config.password)}</password>"
        f"<appKey>{_soap_text(config.app_key)}</appKey>"
        f"<version>{_soap_text(config.version)}</version>"
        f"<style>{_soap_text(config.style)}</style>"
        "</authInfo>"
    )


def _build_talkgroups_request_xml(config: RadioReferenceConfig, req: RadioReferenceTalkgroupRequest) -> str:
    auth_block = _build_auth_block(config)
    tg_cid = req.category_id or 0
    tg_tag = req.tag_id or 0
    tg_dec = req.tgid or 0
    return (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>"
        "<soap:Envelope xmlns:soap=\"http://schemas.xmlsoap.org/soap/envelope/\" "
        "xmlns:tns=\"http://api.radioreference.com/soap2\">"
        "<soap:Body>"
        "<tns:getTrsTalkgroups>"
        f"<sid>{int(req.system_id)}</sid>"
        f"<tgCid>{int(tg_cid)}</tgCid>"
        f"<tgTag>{int(tg_tag)}</tgTag>"
        f"<tgDec>{int(tg_dec)}</tgDec>"
        f"{auth_block}"
        "</tns:getTrsTalkgroups>"
        "</soap:Body>"
        "</soap:Envelope>"
    )


def _text_or_empty(elem: ElementTree.Element | None) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _first_text(elem: ElementTree.Element, tag_names: Iterable[str]) -> str:
    for tag in tag_names:
        found = elem.find(f".//{{*}}{tag}")
        value = _text_or_empty(found)
        if value:
            return value
    return ""


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _has_child_tag(elem: ElementTree.Element, tag_names: Iterable[str]) -> bool:
    wanted = set(tag_names)
    for child in list(elem):
        if _local_name(child.tag) in wanted:
            return True
    return False


def parse_talkgroups_response(xml_text: str) -> dict[int, RadioReferenceTalkgroup]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise RadioReferenceError(f"Failed to parse RadioReference response: {exc}") from exc

    fault = root.find(".//{*}Fault")
    if fault is not None:
        fault_string = _first_text(fault, ["faultstring", "detail"]) or "RadioReference fault"
        raise RadioReferenceError(fault_string)

    talkgroups: dict[int, RadioReferenceTalkgroup] = {}
    for tg_elem in root.iter():
        if not _has_child_tag(tg_elem, ["tgDec", "tgId"]):
            continue
        tg_dec_text = _first_text(tg_elem, ["tgDec", "tgId"])
        if not tg_dec_text:
            continue
        try:
            tgid = int(tg_dec_text)
        except ValueError:
            continue
        if tgid <= 0:
            continue

        name = _first_text(tg_elem, ["tgDescr"]) or f"TG {tgid}"
        alpha = _first_text(tg_elem, ["tgAlpha"])
        subfleet = _first_text(tg_elem, ["tgSubfleet"])
        mode = _first_text(tg_elem, ["tgMode"])

        tag_descr = ""
        for tag in tg_elem.findall(".//{*}tagDescr"):
            text = _text_or_empty(tag)
            if text:
                tag_descr = text
                break

        category = subfleet or tag_descr
        talkgroups[tgid] = RadioReferenceTalkgroup(
            tgid=tgid,
            name=name,
            alpha_tag=alpha,
            category=category,
            mode=mode,
        )

    return talkgroups


def fetch_talkgroups(
    config: RadioReferenceConfig,
    req: RadioReferenceTalkgroupRequest,
) -> dict[int, RadioReferenceTalkgroup]:
    if not config.enabled:
        raise RadioReferenceError("RadioReference integration is disabled")

    payload = _build_talkgroups_request_xml(config, req).encode("utf-8")
    request = urllib.request.Request(
        config.endpoint,
        data=payload,
        headers={
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": "http://api.radioreference.com/soap2#getTrsTalkgroups",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as resp:
            response_text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RadioReferenceError(f"RadioReference HTTP error: {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RadioReferenceError(f"RadioReference connection error: {exc.reason}") from exc

    talkgroups = parse_talkgroups_response(response_text)
    if not talkgroups:
        logger.warning("RadioReference returned no talkgroups for system %s", req.system_id)
    return talkgroups
