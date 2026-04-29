# Epson TM-m30II Status and LED Reference

## Core LEDs
The printer exposes Power, Error, Paper, Wi-Fi, Ethernet, and Bluetooth indicators.

## Useful interpretations
### Power LED
- on: printer powered
- flashing in some cases: startup, power transition, or firmware update context

### Error LED
- lights or flashes when an error occurs
- may also be on briefly after reset before returning online

### Paper LED
- on: out of roll paper
- may flash for user action prompts

### Wi-Fi LED
- on: connected to Wi-Fi
- off: not connected or using wired LAN

### Ethernet LED
- on: connected to Ethernet
- off: not connected or using Wi-Fi

### Bluetooth LED
- on: connected by Bluetooth
- flashing: waiting for pairing

## Status examples
- online: ready for normal printing
- offline: may occur during startup/shutdown, setting modes, paper feed, paper-end, cover-open, or error conditions

## Error boundaries
### Recoverable
- cover open during printing
- slight cutter lock that can be cleared

### Unrecoverable
- high voltage
- CPU execution error
- communication unit error

If unrecoverable error persists after power cycle, escalate.

## Useful maintenance / setup modes
The manual includes:
- self-test mode
- software setting mode
- restore default values mode
- interface setup mode
- status sheet printing

Use these documented modes before guessing advanced fixes.
