# Flashing Red LED on Startup

A flashing red LED on startup is the device's hardware self-test (POST) failure indicator. The flash pattern encodes which subsystem failed.

## Pattern decoder

| Pattern | Meaning | First-line action |
|---|---|---|
| 1 short flash, 1 second pause, repeat | RAM failure | Power-cycle once. If persistent, RMA. |
| 2 short flashes, 1 second pause | eMMC / storage failure | Try recovery mode (hold RESET while powering on). |
| 3 short flashes, 1 second pause | Network interface failure | Disconnect Ethernet/USB peripherals; power-cycle. |
| Continuous fast flash (5 Hz) | Generic POST failure | See "Continuous fast flash" below. |
| Solid red (no flash) | Watchdog timeout — firmware crash | Power-cycle. Capture next failure via `journalctl -b -1`. |

## Continuous fast flash

If you see a continuous fast flash (5 Hz) on startup:

1. Disconnect every cable except power. Power-cycle.
2. If the LED still flashes red, the failure is internal — open a support ticket and include the serial number.
3. If the LED clears with peripherals removed, reconnect them one at a time, power-cycling between each, until the offending peripheral is identified.

## Don't bother with these

- Resetting to factory defaults will not clear a hardware POST failure.
- Updating firmware over the network is not possible — the network interface is not initialised when the LED is in this state.
- Pressing the RESET button does nothing during POST.
