# Meraki MX67 Common Issues

## 1. Device powers on but stays solid orange
### Likely meaning
The appliance has power but is not connected to the Meraki Dashboard.

### Check
- WAN cable path
- upstream Internet connectivity
- whether the device was added to the correct Dashboard network
- license / onboarding completeness
- whether outbound traffic is blocked by an upstream firewall

## 2. Device appears to reboot or never becomes stable after first connection
### Possible cause
Firmware upgrade may be in progress.

### Check
- front LED for flashing white
- allow time for upgrade to complete
- keep power stable during the upgrade window

## 3. Static uplink is needed but the site has no DHCP
### Action
- connect a laptop to a LAN port
- browse to `http://setup.meraki.com`
- configure static IP settings under Local status > Uplink configuration

## 4. Need a second WAN path
### Note
On MX67, LAN2 can be switched from LAN to WAN.

## 5. Reset behavior is unclear
### Reset button behavior
- press about 1 second: delete downloaded configuration and reboot
- press and hold more than 10 seconds: full factory reset

## Escalation boundary
Escalate when:
- Dashboard onboarding is incomplete
- licensing is unknown
- upstream firewall ownership is outside local support scope
- reset does not recover expected behavior
