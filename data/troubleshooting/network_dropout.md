# Intermittent Network Dropouts

Symptom: the device loses network connectivity for 10–60 seconds at irregular intervals, then recovers without intervention. Affects Device Alpha and Device Bravo equally.

## Likely root cause: upstream DHCP lease churn

In ~80% of reported cases, intermittent dropouts trace back to the upstream router renewing the device's DHCP lease while another network event is in flight. The fix:

1. On your router, set the device's DHCP lease time to **at least 24 hours** (default on many consumer routers is 1 hour).
2. Better still, assign a **DHCP reservation** so the device always gets the same IP. This avoids lease renewal entirely from the device's perspective.
3. Power-cycle the router after applying.

## Other causes to rule out, in order

- **Wi-Fi co-channel interference** (Device Bravo only): switch the Wi-Fi radio to 5 GHz from the web console under **Settings → Network → Wi-Fi**.
- **MAC address collision**: if two devices on the LAN have the same MAC (rare but seen with cloned VM images), one will silently drop. Run `arp -a` on a router or admin host to check.
- **Cable**: replace the Ethernet cable with a known-good Cat 5e or better. Rule this out before opening a support ticket.

## What to capture before opening a ticket

If the dropouts persist after a 24-hour lease and a fresh cable, capture:

- The device's `journalctl -k --since "1 hour ago"` for kernel messages.
- A continuous `ping -i 0.2 <gateway>` from the device-side and from a peer host on the same subnet.
- Your router model and the configured Wi-Fi channel + width.
