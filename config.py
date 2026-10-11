"""Configuration settings for the Webcam-Controlled Five-Finger Bionic Hand.

All physical pin mappings, servo angle limits, calibration defaults,
smoothing filters, serial settings, and safety policies are defined here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

# ==============================================================================
# Hardware & Servo Configuration
# ==============================================================================

@dataclass
class ServoFingerConfig:
    name: str
    pin: int                 # Arduino digital PWM/servo pin
    min_safe_angle: int      # Absolute minimum mechanical limit (degrees)
    max_safe_angle: int      # Absolute maximum mechanical limit (degrees)
    open_angle: int          # Target angle when finger is fully OPEN
    closed_angle: int        # Target angle when finger is fully CLOSED
    reversed: bool = False   # True if servo rotates opposite to hand flexion
    max_step_deg: int = 4    # Maximum degrees allowed to move per update tick (slew rate)


# Initial assignments as specified in hardware requirements:
# Thumb: D3, Index: D5, Middle: D6, Ring: D9, Little: D10.
# Note: Keep min_safe_angle and max_safe_angle conservative (e.g. 10° to 170°)
# to protect servo gearboxes and prevent mechanical binding.
DEFAULT_FINGER_CONFIGS: Dict[str, ServoFingerConfig] = {
    "thumb": ServoFingerConfig(
        name="Thumb",
        pin=3,
        min_safe_angle=10,
        max_safe_angle=170,
        open_angle=20,
        closed_angle=160,
        reversed=False,
        max_step_deg=4,
    ),
    "index": ServoFingerConfig(
        name="Index",
        pin=5,
        min_safe_angle=10,
        max_safe_angle=170,
        open_angle=20,
        closed_angle=160,
        reversed=False,
        max_step_deg=5,
    ),
    "middle": ServoFingerConfig(
        name="Middle",
        pin=6,
        min_safe_angle=10,
        max_safe_angle=170,
        open_angle=20,
        closed_angle=160,
        reversed=False,
        max_step_deg=5,
    ),
    "ring": ServoFingerConfig(
        name="Ring",
        pin=9,
        min_safe_angle=10,
        max_safe_angle=170,
        open_angle=20,
        closed_angle=160,
        reversed=False,
        max_step_deg=5,
    ),
    "little": ServoFingerConfig(
        name="Little",
        pin=10,
        min_safe_angle=10,
        max_safe_angle=170,
        open_angle=20,
        closed_angle=160,
        reversed=False,
        max_step_deg=4,
    ),
}

# The ordered list of finger keys used across all serial communication packets:
FINGER_ORDER: List[str] = ["thumb", "index", "middle", "ring", "little"]


# ==============================================================================
# Serial Communication Settings
# ==============================================================================

# Baud rate must match the Arduino sketch
SERIAL_BAUD_RATE: int = 115200

# Timeout for serial read/write in seconds
SERIAL_TIMEOUT: float = 0.05

# Preferred port pattern on macOS: /dev/cu.usbmodem* or /dev/cu.usbserial*
# On Linux: /dev/ttyACM* or /dev/ttyUSB*
PREFERRED_PORT_PATTERNS: List[str] = [
    "/dev/cu.usbmodem*",
    "/dev/cu.usbserial*",
    "/dev/ttyACM*",
    "/dev/ttyUSB*",
]

# Update frequency for transmitting angles to Arduino (Hz)
# 25-35 Hz gives smooth control without flooding Arduino's UART buffer
SERIAL_UPDATE_HZ: float = 30.0
SERIAL_UPDATE_INTERVAL: float = 1.0 / SERIAL_UPDATE_HZ


# ==============================================================================
# Computer Vision & MediaPipe Settings
# ==============================================================================

WEBCAM_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
FPS_TARGET: int = 30

# Path to the MediaPipe Hand Landmarker model file
MODEL_FILE_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_DOWNLOAD_URL: str = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Detection confidence thresholds
MIN_HAND_DETECTION_CONFIDENCE: float = 0.6
MIN_HAND_PRESENCE_CONFIDENCE: float = 0.6
MIN_TRACKING_CONFIDENCE: float = 0.6


# ==============================================================================
# Smoothing, Deadband, & Filtering
# ==============================================================================

# Exponential Moving Average factor (alpha):
# 0.0 = completely frozen, 1.0 = no smoothing (raw instant jump).
# 0.35 provides an ideal balance of responsive feel and jitter suppression.
SMOOTHING_ALPHA: float = 0.35

# Deadband threshold in degrees. If change in target angle is below this value,
# the command is not updated, eliminating tiny sensor noise fluctuations.
ANGLE_DEADBAND_DEG: float = 1.5

# Deadband on normalized bend (0.0 to 1.0)
BEND_DEADBAND: float = 0.015


# ==============================================================================
# Safety & Failsafe Policies
# ==============================================================================

# Time in seconds with no detected hand before triggering lost-tracking policy
LOST_TRACKING_TIMEOUT_SEC: float = 0.45

# Recovery behavior when hand tracking is lost:
# "HOLD" -> Keep the last valid commanded angles
# "REST" -> Gradually transition to the safe rest angles defined below
LOST_TRACKING_ACTION: str = "HOLD"

# Safe rest angles for each finger (used during REST action and initial startup)
SAFE_REST_ANGLES: Dict[str, int] = {
    "thumb": 30,
    "index": 30,
    "middle": 30,
    "ring": 30,
    "little": 30,
}

# Calibration file path
CALIBRATION_FILE_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
