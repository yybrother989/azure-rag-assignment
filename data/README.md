# Corpus Layout

This repo now uses a locked metadata-first corpus contract.

```text
data/
  devices/
    network_access/
      meraki_mx67/
        manuals/
        troubleshooting/
        policies/
    payment_terminal/
      ingenico_desk5000/
        manuals/
        troubleshooting/
        policies/
    check_scanner/
      canon_cr120/
        manuals/
        troubleshooting/
        policies/
    receipt_printer/
      epson_tm_m30ii/
        manuals/
        troubleshooting/
        policies/
  shared/
    manuals/
    troubleshooting/
    policies/
```

## Invariants

1. Only `devices/` and `shared/` exist at the top level.
2. Device identity is always `{device_family}/{model}`.
3. `doc_type` folders are fixed to `manuals`, `troubleshooting`, or `policies`.
4. File names carry topic/version identity, not routing identity.
5. Cross-device material belongs under `shared/`; device-specific material belongs under `devices/`.

## Path Mapping

The ingestion pipeline derives retrieval metadata directly from the path:

- `devices/{device_family}/{model}/{doc_type}/{filename}`
  - `scope=device`
  - `device_family={device_family}`
  - `device={model}`
  - `doc_type={manual|troubleshooting|policy}`
  - `is_shared=false`

- `shared/{doc_type}/{filename}`
  - `scope=shared`
  - `device_family=None`
  - `device=None`
  - `doc_type={manual|troubleshooting|policy}`
  - `is_shared=true`

## Current In-Scope Devices

- `network_access / meraki_mx67`
- `payment_terminal / ingenico_desk5000`
- `check_scanner / canon_cr120`
- `receipt_printer / epson_tm_m30ii`
