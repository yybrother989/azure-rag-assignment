"""
Unit tests for deterministic path metadata parsing in src/extract.py.
"""

from src.extract import _parse_path_metadata


def test_parse_path_metadata_canonical_device_path():
    meta = _parse_path_metadata(
        "devices/network_access/meraki_mx67/manuals/MX67_MX68 Installation Guide.pdf"
    )
    assert meta == {
        "scope": "device",
        "device_family": "network_access",
        "device": "meraki_mx67",
        "doc_type": "manual",
        "topic": "mx67-mx68-installation-guide",
        "version": None,
        "is_shared": False,
    }


def test_parse_path_metadata_shared_path():
    meta = _parse_path_metadata("shared/policies/customer_data_storage.txt")
    assert meta == {
        "scope": "shared",
        "device_family": None,
        "device": None,
        "doc_type": "policy",
        "topic": "customer-data-storage",
        "version": None,
        "is_shared": True,
    }


def test_parse_path_metadata_legacy_device_path():
    meta = _parse_path_metadata("devices/deviceA/manuals/user_manual_v1.2.pdf")
    assert meta == {
        "scope": "device",
        "device_family": None,
        "device": "deviceA",
        "doc_type": "manual",
        "topic": "user-manual",
        "version": "1.2",
        "is_shared": False,
    }
