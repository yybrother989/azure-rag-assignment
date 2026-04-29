# Meraki MX67 Quick Start

## Purpose
Use this guide for first-line setup and basic WAN verification for the Cisco Meraki MX67.

## Before you start
Confirm the following:
- The appliance has been added to the correct Meraki Dashboard network.
- You have the device serial number and license information available.
- You have a wired Internet uplink available for first-time bring-up.
- If an upstream firewall already exists, required outbound connectivity to the Meraki cloud has been allowed.

## Physical overview
The MX67 provides:
- 1 WAN / Internet port
- 4 LAN ports
- 1 USB port
- power input
- reset button

Important note:
- On MX67, **LAN2 can be converted into a second Internet/WAN port** if needed.

## Initial bring-up
1. Connect the MX67 to power.
2. Connect the WAN / Internet port to the upstream circuit.
3. Wait for the device LED to change state.

## LED interpretation
- **Solid orange**: powered, but not connected to Meraki Dashboard
- **Rainbow colors**: attempting to connect to Meraki Dashboard
- **Flashing white**: firmware upgrade in progress
- **Solid white**: fully operational, uplink using wired WAN

## Static IP setup
If DHCP is not available:
1. Connect a laptop to one of the LAN ports.
2. Disable other network services on the laptop if they interfere.
3. Browse to `http://setup.meraki.com`
4. Open **Local status > Uplink configuration**
5. Choose **Static**
6. Enter IP address, subnet mask, gateway, and DNS values
7. Save settings

## DHCP setup
If the upstream network provides DHCP:
- Plug the WAN / Internet port into the upstream circuit
- Wait a few minutes for DHCP negotiation

## Escalate when
Escalate instead of continuing self-debug if:
- the unit never moves past solid orange
- firmware upgrade appears stuck for an unusually long time
- there is no known-good upstream connection
- Dashboard registration or licensing is incomplete
- an upstream firewall policy may be blocking outbound access
