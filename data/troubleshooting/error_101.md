# Error 101 — "Power Supply Out of Range"

## What it means

Error 101 indicates that the device's input voltage is outside the safe operating window during boot or under load. The device's power-management IC has tripped its protection circuit and halted firmware boot to prevent damage.

## Affected devices

- Device Alpha (all firmware revisions)
- Device Bravo running firmware ≥ v3.0.0

## How to fix it

Try the following steps in order. Stop as soon as the error clears.

1. **Verify the power adapter is the one that shipped with the device.** Third-party adapters frequently sag under load. For Device Alpha this is a 12 V / 2 A barrel-jack supply; for Device Bravo, a 24 V PoE++ injector or PSE switch port.

2. **Inspect the cable for damage.** Replace any cable that shows kinks, exposed conductors, or a loose barrel connector. Cable damage is the most common single cause of error 101 in field deployments.

3. **Check the upstream PSU or PoE switch.** Plug the adapter into a different wall outlet, or move the PoE port to a different switch port. Some prosumer PoE switches budget power per port and starve later devices.

4. **Bench-test with a known-good supply.** If you have a spare regulated bench supply, connect it at the rated voltage and confirm boot. If boot succeeds, the original supply or its cable is at fault.

5. **If error 101 persists with a known-good supply and cable**, the device's internal power IC is likely damaged. Open a support ticket with serial number and reproduction steps; out-of-warranty repairs are flat-rate.

## Things that do NOT fix error 101

- Factory reset — error 101 happens before user-space firmware loads.
- Firmware updates — the bootloader trips the protection circuit before the update can apply.
- Removing all attached sensors — error 101 is power-rail-level, not load-induced.
