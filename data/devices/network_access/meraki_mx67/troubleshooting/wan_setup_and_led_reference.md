# Meraki MX67 WAN Setup and LED Reference

## LED status meanings
Use the front status LED as the first diagnostic indicator:

| LED state | Meaning | First action |
|---|---|---|
| Solid orange | Powered but not connected to Dashboard | Check WAN path, upstream Internet, and Dashboard onboarding |
| Rainbow colors | Attempting cloud connection | Wait briefly, then verify Internet path if it does not clear |
| Flashing white | Firmware upgrade in progress | Do not interrupt power unless directed by support |
| Solid white | Fully operational on wired WAN | Basic bring-up successful |
| Solid purple | Cellular failover active | Not typical for a standard MX67 path |

## WAN setup checks
### If using DHCP
- Confirm the WAN port is connected to the correct upstream circuit.
- Wait several minutes for address negotiation.
- Verify that the device can reach the Meraki cloud.

### If using Static IP
Use the local status page:
1. Connect a laptop to a LAN port
2. Browse to `http://setup.meraki.com`
3. Open **Uplink configuration**
4. Enter static IP parameters
5. Save settings

## Secondary WAN note
On the MX67, LAN2 can be reconfigured as a secondary WAN interface.

## Upstream firewall note
If there is already a firewall upstream, outbound connectivity required for Meraki cloud access must be allowed. If the appliance cannot check in, this is a high-probability cause.

## Good operator rule
For first-time setup, prefer:
- wired bring-up first
- firmware completion first
- only then test optional features or alternate uplinks
