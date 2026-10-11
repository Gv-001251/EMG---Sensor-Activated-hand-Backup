#!/usr/bin/env python3
"""Hardware Bringup & Diagnostic Tool for Bionic Hand.

Allows safely verifying the Arduino USB connection and testing individual servos
one-by-one with tendons loose or disconnected, before engaging webcam tracking.

Usage:
  python3 test_hardware.py               # Auto-detects Arduino and tests communication
  python3 test_hardware.py --sweep 0     # Safely sweeps only servo 0 (Thumb) between 30° and 60°
"""

import argparse
import sys
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[!] Error: pyserial is required. Run: pip install pyserial")
    sys.exit(1)

import config


def find_ports():
    ports = serial.tools.list_ports.comports()
    print("Available Serial Ports on macOS:")
    for p in ports:
        print(f"  - {p.device} ({p.description})")
    return [p.device for p in ports]


def probe_arduino(port: str, baud: int = 115200):
    print(f"\n[*] Probing {port} at {baud} baud ...")
    try:
        s = serial.Serial(port, baud, timeout=1.0)
        time.sleep(2.0)  # Wait for Arduino DTR reboot
        s.reset_input_buffer()

        s.write(b"PING\n")
        resp = s.readline().decode("utf-8", errors="ignore").strip()
        print(f"[*] Response to PING: '{resp}'")

        s.write(b"STATUS\n")
        status_resp = s.readline().decode("utf-8", errors="ignore").strip()
        print(f"[*] Response to STATUS: '{status_resp}'")

        s.close()
        if "PONG" in resp or "STATUS" in status_resp or "BIONIC" in resp:
            print(f"[+] SUCCESS: Arduino bionic hand firmware verified on {port}!")
            return True
        else:
            print(f"[!] Arduino responded, but firmware signature was not recognized.")
            print(f"    Please upload arduino/servo_controller.ino to {port}.")
            return False
    except Exception as e:
        print(f"[!] Error opening {port}: {e}")
        return False


def staged_servo_sweep(port: str, finger_idx: int, baud: int = 115200):
    finger_name = config.FINGER_ORDER[finger_idx]
    cfg = config.DEFAULT_FINGER_CONFIGS[finger_name]
    print(f"\n[!] WARNING: Ensure tendons for {cfg.name} (Pin {cfg.pin}) are DISCONNECTED or LOOSE.")
    print(f"[*] Sweeping {cfg.name} servo between 30° and 70° in slow 2° increments ...")
    input("Press ENTER to proceed or Ctrl+C to cancel...")

    try:
        s = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2.0)
        s.reset_input_buffer()

        # Sweep forward 30 -> 70
        for ang in range(30, 71, 2):
            cmd = f"<TEST,{finger_idx},{ang}>\n".encode("utf-8")
            s.write(cmd)
            time.sleep(0.08)

        time.sleep(0.5)

        # Sweep backward 70 -> 30
        for ang in range(70, 29, -2):
            cmd = f"<TEST,{finger_idx},{ang}>\n".encode("utf-8")
            s.write(cmd)
            time.sleep(0.08)

        # Return to rest
        s.write(b"<REST>\n")
        time.sleep(0.5)
        s.close()
        print(f"[+] Sweep for {cfg.name} completed successfully.")
    except Exception as e:
        print(f"[!] Error during sweep: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hardware Bringup Tool for Bionic Hand")
    parser.add_argument("--port", type=str, default="/dev/cu.usbmodem101", help="Serial port to probe")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--sweep", type=int, default=None, choices=[0, 1, 2, 3, 4], help="Sweep single finger index (0=Thumb..4=Little)")
    parser.add_argument("--list", action="store_true", help="List available serial ports and exit")

    args = parser.parse_args()

    if args.list:
        find_ports()
        return

    find_ports()
    if args.sweep is not None:
        staged_servo_sweep(args.port, args.sweep, args.baud)
    else:
        probe_arduino(args.port, args.baud)


if __name__ == "__main__":
    main()
