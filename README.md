# Webcam-Controlled Five-Finger Bionic Hand

A real-time computer-vision application that controls a 5-servo bionic hand using a MacBook webcam, MediaPipe 21-point hand tracking, and an Arduino Uno over USB Serial.

---

## 1. System Architecture

```
┌─────────────────┐       MediaPipe 21-pt        ┌─────────────────────────┐
│  MacBook Webcam │ ───────────────────────────> │     webcam_hand.py      │
└─────────────────┘      (60 FPS capture)        │  - Kinematic Extraction │
                                                 │  - Custom Calibration   │
                                                 │  - EMA Smoothing/Filter │
                                                 │  - Safety & E-STOP      │
                                                 └───────────┬─────────────┘
                                                             │ Framed Serial Packet:
                                                             │ <T,I,M,R,L>\n (115200 baud)
                                                             v
┌─────────────────────────┐     PWM Signals      ┌─────────────────────────┐
│     5 Servo Motors      │ <─────────────────── │       Arduino Uno       │
│ Thumb:  D3   Ring:  D9  │  Non-blocking slew   │ - Packet Validator      │
│ Index:  D5   Pinky: D10 │  rate angle updates  │ - Mechanical Clamping   │
│ Middle: D6              │  (50 Hz tick loop)   │ - Watchdog Failsafe     │
└────────────┬────────────┘                      └─────────────────────────┘
             │
   External Power Supply (5V–6V, ≥3A)
   Common Ground with Arduino Uno
```

---

## 2. Hardware Wiring & Power Requirements

> [!CAUTION]
> **NEVER power the 5 servos directly from the Arduino Uno's 5V pin!**
> Five servos under load draw 1.5A to 3.5A of stall current. Attempting to power them from the Arduino board will cause brownout resets, USB disconnection, or permanent damage to your MacBook USB port.

### Power Wiring
1. **External Power Supply:** Connect external 5V–6V (≥3A regulated) **(+)** to the red (power) wires of all five servos.
2. **Common Ground:** Connect external power **GND (-)** to:
   - The brown/black (ground) wires of all five servos.
   - Any **GND pin** on the Arduino Uno.
3. **USB Connection:** Connect the Arduino Uno to your MacBook via standard USB cable (this powers the Arduino microcontroller).

### Servo Pin Assignment (Configurable in `config.py` and `arduino/servo_controller.ino`)
| Finger | Arduino Uno Pin | Timer / Type | Default Safe Min | Default Safe Max |
| :--- | :--- | :--- | :--- | :--- |
| **Thumb** | `Digital Pin 3` | PWM | 10° | 170° |
| **Index** | `Digital Pin 5` | PWM | 10° | 170° |
| **Middle** | `Digital Pin 6` | PWM | 10° | 170° |
| **Ring** | `Digital Pin 9` | PWM | 10° | 170° |
| **Little (Pinky)** | `Digital Pin 10` | PWM | 10° | 170° |

---

## 3. macOS Setup Guide (Step-by-Step)

### Step 3.1: Python Virtual Environment Setup
Open Terminal on your Mac and navigate to this project folder:
```bash
cd /Users/ganesan.kgayathri/Documents/PROJECTS/Hand

# 1. Create a Python 3 virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install project dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3.2: Verify Serial Port Discovery
Connect the Arduino Uno to your Mac via USB, then list available USB serial devices:
```bash
ls -la /dev/cu.usb*
```
On macOS, your Arduino Uno appears as `/dev/cu.usbmodem101` (or similar number like `usbmodem14101`).

You can also run the built-in diagnostic port finder:
```bash
python3 test_hardware.py --list
```

### Step 3.3: Upload Firmware to Arduino Uno

#### Option A: Using Arduino IDE (Recommended GUI)
1. Launch **Arduino IDE** from `/Applications`.
2. Go to **File -> Open...** and open `arduino/servo_controller/servo_controller.ino`.
3. Under **Tools -> Board**, select **Arduino Uno**.
4. Under **Tools -> Port**, select `/dev/cu.usbmodem101` (or your detected Arduino port).
5. Ensure the **Servo** library is installed (Go to **Tools -> Manage Libraries...**, search for `Servo`, and click Install if prompted).
6. Click the **Upload** button (arrow icon).

#### Option B: Using Command Line (`arduino-cli`)
```bash
# Compile the sketch
"/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli" compile \
  -b arduino:avr:uno arduino/servo_controller

# Upload to Arduino Uno
"/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli" upload \
  -b arduino:avr:uno -p /dev/cu.usbmodem101 arduino/servo_controller
```

### Step 3.4: Grant macOS Camera Permission
When running the Python application for the first time, macOS will prompt to grant Camera access to Terminal (or VS Code).
If the webcam feed does not appear:
1. Open **System Settings -> Privacy & Security -> Camera**.
2. Ensure your terminal app (Terminal, iTerm, or Cursor/VS Code) is toggled **ON**.

---

## 4. Safe Initial Hardware Bringup & Staged Testing

> [!IMPORTANT]
> **INITIAL BRINGUP PROCEDURE (DO NOT SKIP):**
> 1. Disconnect or loosen all finger tendons before first power-up.
> 2. Verify servo rotation directions before stringing mechanical linkages.
> 3. Identify each physical servo individually using Staged Test Mode.

### Step 4.1: Probe Arduino Communication
Test that the Arduino firmware responds correctly over serial:
```bash
python3 test_hardware.py --port /dev/cu.usbmodem101
```
Expected output:
```
[*] Response to PING: 'PONG'
[*] Response to STATUS: 'STATUS:RUNNING,CUR=30/30/30/30/30,TGT=30/30/30/30/30'
[+] SUCCESS: Arduino bionic hand firmware verified on /dev/cu.usbmodem101!
```

### Step 4.2: Test Each Servo Individually
Sweep single servos slowly between 30° and 70° to verify wiring and identify which motor controls which finger:
```bash
# Test Thumb (Index 0, Pin 3)
python3 test_hardware.py --sweep 0

# Test Index Finger (Index 1, Pin 5)
python3 test_hardware.py --sweep 1

# Test Middle Finger (Index 2, Pin 6)
python3 test_hardware.py --sweep 2

# Test Ring Finger (Index 3, Pin 9)
python3 test_hardware.py --sweep 3

# Test Little Finger (Index 4, Pin 10)
python3 test_hardware.py --sweep 4
```

---

## 5. Launching the Webcam Application

### Launch with Real Hardware (Auto-detects USB port)
```bash
python3 webcam_hand.py
```

### Launch in Simulation Mode (No Arduino required)
```bash
python3 webcam_hand.py --simulate
```

### Launch Directly in Staged Hardware Test Mode
```bash
python3 webcam_hand.py --staged-test
```

### Custom Serial Port or Camera Index
```bash
python3 webcam_hand.py --port /dev/cu.usbmodem101 --cam 0
```

---

## 6. Live Controls & Calibration

### Calibration Controls
Because everyone's hand size, camera distance, and natural reach differ, the application uses **custom calibration** rather than hardcoded pixel thresholds:
1. Hold your hand in front of the camera with all 5 fingers **completely extended open**.
   - Press **`O`** to record the **Open Hand** baseline.
2. Form a **tight fist** with all fingers and thumb curled inward.
   - Press **`C`** to record the **Closed Hand** baseline.
3. Press **`S`** to save your calibration to `calibration.json`. (It loads automatically on future launches).
4. Press **`R`** to reset to default factory references.

### Safety & Emergency Stop (E-STOP)
- **`SPACE` or `E`**: Immediately toggles **EMERGENCY STOP**.
  - Sends a `<STOP>` packet to the Arduino.
  - Halts all new motion command transmission.
  - Flashes a high-visibility warning banner across the video screen.
  - *Note:* Software E-STOP freezes servo target updates immediately; it does not physically cut DC VCC power to the motors. For hardware isolation, disconnect the external power supply switch.
- **`SPACE` or `E` again**: Clears E-STOP and safely resumes tracking.

### Staged Test Mode (During Webcam Operation)
- Press **`T`** to toggle Staged Test Mode.
- Press **`1`**, **`2`**, **`3`**, **`4`**, or **`5`** to select Thumb, Index, Middle, Ring, or Little.
- Press **`+` / `W`** or **`-` / `S`** to step that servo's angle by ±5° without moving any other finger.

### Additional Keyboard Shortcuts
- **`Q`** or **`ESC`**: Safely returns all servos to safe rest angles (30°) and exits cleanly.

---

## 7. Testing & Verification

### Automated Unit Tests (No hardware required)
Run the 18 automated unit and integration tests:
```bash
python3 -m unittest test_hand_controller.py -v
```
Tests cover:
- Pin assignments and configuration validity
- 3D vector geometry and bone ratio kinematics
- Open hand vs. fist metric calculations
- Calibration recording, calculation, and JSON persistence
- Servo angle mapping, direction inversion, EMA smoothing, and deadband
- Loss-of-tracking timeout and Emergency Stop state machine
- Full simulation of the Arduino C++ framed packet parser

### Hardware Integration Tests (Requires Arduino & Servos)
- Run `python3 test_hardware.py --port /dev/cu.usbmodem101` to verify USB link.
- Run `python3 test_hardware.py --sweep 0` to verify physical servo movement.

---

## 8. Configuration Customization (`config.py`)

You can customize all hardware and software parameters in `config.py`:
- **Inverting Servo Direction:** Set `reversed=True` on any finger if its tendon pulls when the servo turns counter-clockwise.
- **Changing Angle Limits:** Adjust `min_safe_angle` and `max_safe_angle` per finger to prevent pulling tendons past mechanical end-stops.
- **Smoothing Factor:** Change `SMOOTHING_ALPHA` (default `0.35`). Lower values increase smoothing (slower response); higher values reduce lag (more responsive).
- **Lost Tracking Behavior:** Set `LOST_TRACKING_ACTION = "HOLD"` to maintain position when hand leaves webcam view, or `"REST"` to return gently to 30°.
- **Serial Transmission Rate:** Set `SERIAL_UPDATE_HZ = 30.0` (30 updates per second).
