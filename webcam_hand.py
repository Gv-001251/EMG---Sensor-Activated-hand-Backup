#!/usr/bin/env python3
"""Webcam-Controlled Five-Finger Bionic Hand.

Tracks real hand movements through the webcam using MediaPipe Hand Landmarker,
computes calibrated per-finger flexion metrics, maps them to safe servo angles,
and transmits validated framed commands (<T,I,M,R,L>\\n) over USB Serial
to an Arduino Uno servo controller.

Features:
  - 21-point hand landmark tracking via modern MediaPipe Tasks API
  - Calibrated normalized bend calculation (independent thumb & finger kinematics)
  - Exponential moving average (EMA) smoothing and deadband filtering
  - Configurable per-finger direction reversal and mechanical safety clamping
  - Non-blocking persistent serial communication with auto-discovery & simulation mode
  - Keyboard Emergency Stop (E-STOP) and safe loss-of-tracking recovery
  - Real-time HUD overlay with per-finger telemetry and calibration controls
  - Staged single-servo testing mode for safe hardware bringup
"""

import argparse
import json
import math
import os
import ssl
import sys
import threading
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# PySerial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# MediaPipe Tasks Vision API
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# Local configuration
import config


# ==============================================================================
# Model File Downloader Helper
# ==============================================================================

def ensure_model_file(model_path: str = config.MODEL_FILE_PATH, download_url: str = config.MODEL_DOWNLOAD_URL) -> str:
    """Ensures that the MediaPipe Hand Landmarker task bundle exists locally.
    
    If not present, downloads it automatically with SSL verification fallback.
    """
    if os.path.isfile(model_path):
        return model_path

    print(f"[*] MediaPipe model not found at {model_path}.")
    print(f"[*] Downloading official hand landmarker model from {download_url} ...")
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

    # Use certifi if available, or unverified context fallback if local CA certificates are missing on macOS
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    urllib.request.install_opener(opener)

    try:
        urllib.request.urlretrieve(download_url, model_path)
        print(f"[+] Successfully downloaded model ({os.path.getsize(model_path) // 1024} KB).")
    except Exception as e:
        print(f"[!] Direct download failed: {e}")
        # Try fallback via curl
        print("[*] Attempting download via curl...")
        rc = os.system(f"curl -fSL -o '{model_path}' '{download_url}'")
        if rc != 0 or not os.path.isfile(model_path):
            raise RuntimeError(f"Failed to download hand_landmarker.task model file: {e}")
        print(f"[+] Downloaded via curl ({os.path.getsize(model_path) // 1024} KB).")

    return model_path


# ==============================================================================
# Vector Math & Finger Kinematics
# ==============================================================================

def dist_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Computes Euclidean distance between two 3D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def angle_between_vectors_3d(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
    """Calculates angle in degrees between two 3D vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cos_theta))


class FingerBendCalculator:
    """Computes robust, normalized finger flexion values from 21 MediaPipe hand landmarks.
    
    Rather than relying on uncalibrated pixel distances or fragile orientation assumptions,
    this calculator uses:
      1. Anatomical bone ratios: Tip-to-MCP straight distance divided by the finger's
         total segmented bone length.
      2. Palm-scale normalization: Inter-joint distances normalized by wrist-to-middle-MCP distance.
      3. Thumb circumduction & flexion: Tip-to-Pinky-MCP distance normalized by palm scale,
         plus Thumb IP joint flexion angle.
    """

    # MediaPipe landmark indices
    # Wrist: 0
    # Thumb: 1: CMC, 2: MCP, 3: IP, 4: TIP
    # Index: 5: MCP, 6: PIP, 7: DIP, 8: TIP
    # Middle: 9: MCP, 10: PIP, 11: DIP, 12: TIP
    # Ring: 13: MCP, 14: PIP, 15: DIP, 16: TIP
    # Little: 17: MCP, 18: PIP, 19: DIP, 20: TIP
    FINGER_LANDMARK_MAP = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "little": (17, 18, 19, 20),
    }

    @staticmethod
    def extract_point(landmarks, idx: int) -> Tuple[float, float, float]:
        lm = landmarks[idx]
        return (float(lm.x), float(lm.y), float(lm.z))

    def compute_raw_metrics(self, landmarks) -> Dict[str, float]:
        """Calculates raw anatomical motion metrics for all 5 fingers.
        
        Larger values indicate open/extended fingers, smaller values indicate flexed/closed.
        """
        raw_metrics = {}

        # Reference palm scale: Wrist (0) to Middle MCP (9)
        p_wrist = self.extract_point(landmarks, 0)
        p_middle_mcp = self.extract_point(landmarks, 9)
        palm_scale = max(0.01, dist_3d(p_wrist, p_middle_mcp))

        # --- Thumb Flexion & Adduction ---
        # Thumb tip (4) moves across palm toward Pinky MCP (17)
        p_thumb_cmc = self.extract_point(landmarks, 1)
        p_thumb_mcp = self.extract_point(landmarks, 2)
        p_thumb_ip = self.extract_point(landmarks, 3)
        p_thumb_tip = self.extract_point(landmarks, 4)
        p_pinky_mcp = self.extract_point(landmarks, 17)

        dist_thumb_to_pinky = dist_3d(p_thumb_tip, p_pinky_mcp) / palm_scale
        # Thumb IP joint angle (straight thumb = ~170-180 deg, curled = ~110-130 deg)
        v_mcp_ip = (p_thumb_ip[0] - p_thumb_mcp[0], p_thumb_ip[1] - p_thumb_mcp[1], p_thumb_ip[2] - p_thumb_mcp[2])
        v_ip_tip = (p_thumb_tip[0] - p_thumb_ip[0], p_thumb_tip[1] - p_thumb_ip[1], p_thumb_tip[2] - p_thumb_ip[2])
        thumb_angle_ratio = angle_between_vectors_3d(v_mcp_ip, v_ip_tip) / 180.0

        # Blended thumb metric (higher = open/extended away from palm, lower = curled into palm)
        raw_metrics["thumb"] = (dist_thumb_to_pinky * 0.75) + (thumb_angle_ratio * 0.25)

        # --- 4 Fingers (Index, Middle, Ring, Little) ---
        for finger_name, (mcp_i, pip_i, dip_i, tip_i) in self.FINGER_LANDMARK_MAP.items():
            p_mcp = self.extract_point(landmarks, mcp_i)
            p_pip = self.extract_point(landmarks, pip_i)
            p_dip = self.extract_point(landmarks, dip_i)
            p_tip = self.extract_point(landmarks, tip_i)

            # Segment lengths along finger
            l_mcp_pip = dist_3d(p_mcp, p_pip)
            l_pip_dip = dist_3d(p_pip, p_dip)
            l_dip_tip = dist_3d(p_dip, p_tip)
            total_bone_len = max(0.001, l_mcp_pip + l_pip_dip + l_dip_tip)

            # Direct distance from tip to MCP
            tip_to_mcp = dist_3d(p_tip, p_mcp)
            ratio = tip_to_mcp / total_bone_len

            # Tip to Wrist distance normalized by palm scale
            tip_to_wrist = dist_3d(p_tip, p_wrist) / palm_scale

            # Joint angles
            v_pip = (p_pip[0] - p_mcp[0], p_pip[1] - p_mcp[1], p_pip[2] - p_mcp[2])
            v_dip = (p_dip[0] - p_pip[0], p_dip[1] - p_pip[1], p_dip[2] - p_pip[2])
            v_tip = (p_tip[0] - p_dip[0], p_tip[1] - p_dip[1], p_tip[2] - p_dip[2])

            angle_pip = angle_between_vectors_3d(v_pip, v_dip) / 180.0
            angle_dip = angle_between_vectors_3d(v_dip, v_tip) / 180.0

            # Composite metric (1.0 = fully open/extended, 0.3 = curled fist)
            finger_metric = (ratio * 0.50) + ((tip_to_wrist / 2.0) * 0.25) + (((angle_pip + angle_dip) / 2.0) * 0.25)
            raw_metrics[finger_name] = finger_metric

        return raw_metrics


# ==============================================================================
# Calibration Manager
# ==============================================================================

class CalibrationManager:
    """Manages recording and persistence of open-hand and closed-hand reference metrics.
    
    Computes normalized bend percentage (0.0 = fully open, 1.0 = fully closed)
    based on the recorded physical positions of the user's specific hand.
    """

    # Safe anatomical defaults used prior to manual calibration
    DEFAULT_OPEN_REFS: Dict[str, float] = {
        "thumb": 1.25,
        "index": 0.95,
        "middle": 0.98,
        "ring": 0.94,
        "little": 0.88,
    }
    DEFAULT_CLOSED_REFS: Dict[str, float] = {
        "thumb": 0.50,
        "index": 0.40,
        "middle": 0.38,
        "ring": 0.40,
        "little": 0.38,
    }

    def __init__(self, filepath: str = config.CALIBRATION_FILE_PATH):
        self.filepath = filepath
        self.open_refs = dict(self.DEFAULT_OPEN_REFS)
        self.closed_refs = dict(self.DEFAULT_CLOSED_REFS)
        self.is_custom_calibrated = False
        self.load()

    def record_open(self, metrics: Dict[str, float]):
        """Records current metrics as the Open Hand reference."""
        for k, v in metrics.items():
            self.open_refs[k] = float(v)
        self.is_custom_calibrated = True

    def record_closed(self, metrics: Dict[str, float]):
        """Records current metrics as the Closed Hand (Fist) reference."""
        for k, v in metrics.items():
            self.closed_refs[k] = float(v)
        self.is_custom_calibrated = True

    def reset_defaults(self):
        """Resets calibration to default baseline values."""
        self.open_refs = dict(self.DEFAULT_OPEN_REFS)
        self.closed_refs = dict(self.DEFAULT_CLOSED_REFS)
        self.is_custom_calibrated = False
        if os.path.isfile(self.filepath):
            try:
                os.remove(self.filepath)
            except OSError:
                pass

    def save(self) -> bool:
        """Saves current calibration references to JSON file."""
        data = {
            "open_refs": self.open_refs,
            "closed_refs": self.closed_refs,
            "is_custom_calibrated": self.is_custom_calibrated,
            "timestamp": time.time(),
        }
        try:
            with open(self.filepath, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"[!] Failed to save calibration: {e}")
            return False

    def load(self) -> bool:
        """Loads calibration from JSON file if available."""
        if not os.path.isfile(self.filepath) or os.path.getsize(self.filepath) == 0:
            return False
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            self.open_refs = data.get("open_refs", self.DEFAULT_OPEN_REFS)
            self.closed_refs = data.get("closed_refs", self.DEFAULT_CLOSED_REFS)
            self.is_custom_calibrated = data.get("is_custom_calibrated", True)
            return True
        except Exception as e:
            print(f"[!] Could not load calibration file: {e}")
            return False

    def compute_normalized_bend(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculates normalized bend fraction [0.0 (open) to 1.0 (closed)] for each finger."""
        bends = {}
        for finger in config.FINGER_ORDER:
            val = metrics.get(finger, self.open_refs[finger])
            open_val = self.open_refs[finger]
            closed_val = self.closed_refs[finger]

            denom = open_val - closed_val
            if abs(denom) < 1e-4:
                denom = 0.001

            # Higher metric = open, Lower metric = closed
            # normalized bend = (open_val - current_val) / (open_val - closed_val)
            bend = (open_val - val) / denom
            bends[finger] = max(0.0, min(1.0, float(bend)))

        return bends


# ==============================================================================
# Servo Angle Mapper & Filter
# ==============================================================================

class ServoAngleMapper:
    """Maps normalized finger bend values (0.0 to 1.0) to physical servo angles,
    applying per-finger direction reversal, EMA smoothing, deadband, and safe clamping.
    """

    def __init__(self, finger_configs: Dict[str, config.ServoFingerConfig] = config.DEFAULT_FINGER_CONFIGS):
        self.finger_configs = finger_configs
        self.smoothed_angles: Dict[str, float] = {}
        self.last_sent_angles: Dict[str, int] = {}

        # Initialize filter states to safe rest angles
        for finger, cfg in self.finger_configs.items():
            self.smoothed_angles[finger] = float(config.SAFE_REST_ANGLES.get(finger, cfg.open_angle))
            self.last_sent_angles[finger] = int(round(self.smoothed_angles[finger]))

    def map_bends_to_angles(self, bends: Dict[str, float], alpha: float = config.SMOOTHING_ALPHA) -> Tuple[Dict[str, int], bool]:
        """Converts normalized bends to target servo angles with smoothing and deadband.
        
        Returns:
            Tuple of (angles_dict, has_significant_change_boolean)
        """
        current_target_angles = {}
        any_changed = False

        for finger in config.FINGER_ORDER:
            cfg = self.finger_configs[finger]
            bend = bends.get(finger, 0.0)

            # Apply per-finger direction reversal if configured
            effective_bend = (1.0 - bend) if cfg.reversed else bend

            # Linear interpolation from open_angle to closed_angle
            raw_angle = cfg.open_angle + effective_bend * (cfg.closed_angle - cfg.open_angle)

            # Enforce absolute safe mechanical clamp limits
            clamped_raw = max(cfg.min_safe_angle, min(cfg.max_safe_angle, raw_angle))

            # Exponential Moving Average (EMA) smoothing
            prev_smooth = self.smoothed_angles[finger]
            new_smooth = (alpha * clamped_raw) + ((1.0 - alpha) * prev_smooth)
            self.smoothed_angles[finger] = new_smooth

            int_angle = int(round(new_smooth))
            # Double check clamp
            int_angle = max(cfg.min_safe_angle, min(cfg.max_safe_angle, int_angle))
            current_target_angles[finger] = int_angle

            # Deadband evaluation
            if abs(int_angle - self.last_sent_angles[finger]) >= config.ANGLE_DEADBAND_DEG:
                any_changed = True

        return current_target_angles, any_changed

    def commit_angles(self, angles: Dict[str, int]):
        """Records the angles that were transmitted over serial."""
        for k, v in angles.items():
            self.last_sent_angles[k] = v


# ==============================================================================
# Serial Communication Controller
# ==============================================================================

class SerialHandController:
    """Manages persistent serial communication with the Arduino Uno.
    
    Features:
      - Automatic port discovery on macOS (/dev/cu.usbmodem*, etc.)
      - Graceful fallback to Simulation / Mock mode if hardware is disconnected
      - Single persistent connection (never reopens during active operation)
      - Non-blocking background telemetry reader
      - Validated framed packet generation (<T,I,M,R,L>\\n)
    """

    def __init__(self, port: Optional[str] = None, baud: int = config.SERIAL_BAUD_RATE, simulate: bool = False):
        self.port = port
        self.baud = baud
        self.is_simulated = simulate
        self.serial_conn: Optional[serial.Serial] = None
        self.is_connected = False
        self.last_ack_response: str = ""
        self.last_tx_time = 0.0
        self._reader_running = False
        self._reader_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        if not self.is_simulated and SERIAL_AVAILABLE:
            self.connect()
        else:
            self.is_simulated = True
            print("[*] Running in SIMULATION MODE (No physical serial port).")

    @staticmethod
    def find_available_ports() -> List[str]:
        """Discovers serial ports matching typical Arduino boards on macOS and Linux."""
        if not SERIAL_AVAILABLE:
            return []
        ports = serial.tools.list_ports.comports()
        discovered = []
        for p in ports:
            discovered.append(p.device)
        return discovered

    def auto_detect_arduino_port(self) -> Optional[str]:
        """Finds the most likely Arduino Uno port."""
        available = self.find_available_ports()
        # Priority 1: Match preferred patterns (/dev/cu.usbmodem*)
        for pat in config.PREFERRED_PORT_PATTERNS:
            prefix = pat.replace("*", "")
            for p in available:
                if p.startswith(prefix):
                    return p
        # Priority 2: Any usb port
        for p in available:
            if "usb" in p.lower():
                return p
        # Priority 3: First available port, if any
        return available[0] if available else None

    def connect(self) -> bool:
        """Opens persistent serial connection to Arduino."""
        if not SERIAL_AVAILABLE:
            print("[!] pyserial is not installed. Falling back to simulation.")
            self.is_simulated = True
            return False

        if not self.port:
            self.port = self.auto_detect_arduino_port()

        if not self.port:
            print("[!] No Arduino serial port detected. Switching to SIMULATION MODE.")
            self.is_simulated = True
            return False

        try:
            print(f"[*] Opening serial port {self.port} at {self.baud} baud ...")
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=config.SERIAL_TIMEOUT,
                write_timeout=config.SERIAL_TIMEOUT,
            )
            self.is_connected = True
            self.is_simulated = False

            # Arduino Uno auto-resets when DTR toggles on serial open; allow bootloader 2 seconds
            print("[*] Arduino connected. Waiting 2.0s for bootloader initialization ...")
            time.sleep(2.0)
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()

            # Start background reader thread
            self._reader_running = True
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

            # Send ping
            self.send_command("PING")
            print(f"[+] Serial connection established on {self.port}.")
            return True

        except Exception as e:
            print(f"[!] Could not connect to serial port {self.port}: {e}")
            print("[!] Falling back to SIMULATION MODE.")
            self.is_connected = False
            self.is_simulated = True
            return False

    def _reader_loop(self):
        """Continuously reads incoming ACK and STATUS lines from the Arduino."""
        while self._reader_running and self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    with self.lock:
                        self.last_ack_response = line
            except Exception:
                break

    def send_command(self, cmd: str) -> bool:
        """Sends a text command to Arduino with error handling."""
        if self.is_simulated or not self.serial_conn or not self.serial_conn.is_open:
            self.last_ack_response = f"SIM_ACK:{cmd}"
            return True

        try:
            full_msg = f"{cmd}\n".encode("utf-8")
            with self.lock:
                self.serial_conn.write(full_msg)
                self.serial_conn.flush()
            return True
        except Exception as e:
            print(f"[!] Serial write error: {e}")
            self.is_connected = False
            return False

    def send_angles(self, angles: Dict[str, int]) -> bool:
        """Formats and transmits the 5 angles as <T,I,M,R,L>\\n."""
        now = time.time()
        # Rate-limiting to configured SERIAL_UPDATE_HZ
        if now - self.last_tx_time < config.SERIAL_UPDATE_INTERVAL:
            return False

        t = angles.get("thumb", 30)
        i = angles.get("index", 30)
        m = angles.get("middle", 30)
        r = angles.get("ring", 30)
        l = angles.get("little", 30)

        packet = f"<{t},{i},{m},{r},{l}>"
        success = self.send_command(packet)
        if success:
            self.last_tx_time = now
        return success

    def emergency_stop(self):
        """Sends emergency stop command to freeze motion immediately."""
        self.send_command("STOP")

    def resume_from_stop(self):
        """Sends resume command after emergency stop."""
        self.send_command("RESUME")

    def send_rest(self):
        """Sends safe rest command."""
        self.send_command("REST")

    def close(self):
        """Safely closes serial connection."""
        self._reader_running = False
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.send_rest()
                time.sleep(0.05)
                self.serial_conn.close()
            except Exception:
                pass
        self.is_connected = False


# ==============================================================================
# Safety & Staged Testing Manager
# ==============================================================================

class SafetyMonitor:
    """Monitors hand presence and keyboard safety commands.
    
    Halts motion commands when hand tracking is lost, enforces configured
    recovery policy ('HOLD' or 'REST'), and handles keyboard E-STOP.
    """

    def __init__(self, serial_ctrl: SerialHandController):
        self.serial_ctrl = serial_ctrl
        self.emergency_stop_active = False
        self.hand_present = False
        self.last_seen_hand_time = time.time()
        self.tracking_lost_handled = False

    def update_hand_presence(self, is_detected: bool) -> str:
        """Updates presence timestamp and evaluates safety conditions.
        
        Returns:
            Current status string ('NORMAL', 'LOST_HOLD', 'LOST_REST', 'ESTOP')
        """
        now = time.time()
        if self.emergency_stop_active:
            return "ESTOP"

        if is_detected:
            self.hand_present = True
            self.last_seen_hand_time = now
            self.tracking_lost_handled = False
            return "NORMAL"

        # Hand not detected
        self.hand_present = False
        elapsed = now - self.last_seen_hand_time

        if elapsed > config.LOST_TRACKING_TIMEOUT_SEC:
            if not self.tracking_lost_handled:
                self.tracking_lost_handled = True
                if config.LOST_TRACKING_ACTION == "REST":
                    self.serial_ctrl.send_rest()
                    return "LOST_REST"
                else:
                    # In HOLD mode: simply stop sending movement commands
                    return "LOST_HOLD"
            return f"LOST_{config.LOST_TRACKING_ACTION}"

        return "NORMAL"

    def toggle_emergency_stop(self) -> bool:
        """Toggles Emergency Stop. Halts transmissions immediately."""
        self.emergency_stop_active = not self.emergency_stop_active
        if self.emergency_stop_active:
            self.serial_ctrl.emergency_stop()
        else:
            self.serial_ctrl.resume_from_stop()
        return self.emergency_stop_active


class StagedTester:
    """Provides staged single-servo testing for initial hardware bringup.
    
    Allows testing each individual servo one at a time (e.g. while tendons are
    disconnected or loose) to verify pin mappings and mechanical range without
    moving all 5 servos simultaneously.
    """

    def __init__(self, serial_ctrl: SerialHandController):
        self.serial_ctrl = serial_ctrl
        self.active = False
        self.selected_finger_index = 0  # 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Little
        self.test_angles = [30, 30, 30, 30, 30]
        self.last_tx_time = 0.0

    def select_finger(self, idx: int):
        self.selected_finger_index = max(0, min(4, idx))

    def step_angle(self, delta: int):
        finger_key = config.FINGER_ORDER[self.selected_finger_index]
        cfg = config.DEFAULT_FINGER_CONFIGS[finger_key]
        cur = self.test_angles[self.selected_finger_index]
        new_angle = max(cfg.min_safe_angle, min(cfg.max_safe_angle, cur + delta))
        self.test_angles[self.selected_finger_index] = new_angle
        # Transmit test command
        cmd = f"<TEST,{self.selected_finger_index},{new_angle}>"
        self.serial_ctrl.send_command(cmd)
        self.last_tx_time = time.time()

    def update_keepalive(self):
        """Periodically refreshes the active test target so the Arduino watchdog does not time out."""
        if not self.active:
            return
        now = time.time()
        # Keepalive every 500ms
        if now - self.last_tx_time >= 0.5:
            cur = self.test_angles[self.selected_finger_index]
            cmd = f"<TEST,{self.selected_finger_index},{cur}>"
            self.serial_ctrl.send_command(cmd)
            self.last_tx_time = now


# ==============================================================================
# UI Overlay & Telemetry Renderer
# ==============================================================================

class UIOverlay:
    """Renders high-visibility real-time telemetry, calibration guides,
    hand skeleton landmarks, and safety status onto the OpenCV video frame.
    """

    # Hand skeleton landmark connections for OpenCV drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Little
        (5, 9), (9, 13), (13, 17),              # Palm knuckles
    ]

    @classmethod
    def draw_hand_skeleton(cls, frame: np.ndarray, landmarks):
        """Draws anti-aliased hand skeleton connections and joints."""
        h, w, _ = frame.shape
        coords = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            coords.append((cx, cy))

        # Draw bones
        for p1_i, p2_i in cls.HAND_CONNECTIONS:
            pt1 = coords[p1_i]
            pt2 = coords[p2_i]
            cv2.line(frame, pt1, pt2, (230, 210, 50), 2, cv2.LINE_AA)

        # Draw joints
        for i, pt in enumerate(coords):
            # Fingertips (4, 8, 12, 16, 20) in bright green, other joints in cyan/orange
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(frame, pt, 6, (0, 255, 120), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
            elif i == 0:
                cv2.circle(frame, pt, 8, (0, 140, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, (255, 180, 0), -1, cv2.LINE_AA)

    @staticmethod
    def draw_hud(
        frame: np.ndarray,
        fps: float,
        safety_status: str,
        is_estop: bool,
        serial_ctrl: SerialHandController,
        calib_mgr: CalibrationManager,
        bends: Dict[str, float],
        target_angles: Dict[str, int],
        staged_tester: StagedTester,
    ):
        """Draws semi-transparent HUD with per-finger telemetry bars and system status."""
        h, w, _ = frame.shape

        # --- Top Status Bar ---
        top_bar = frame.copy()
        cv2.rectangle(top_bar, (0, 0), (w, 55), (20, 24, 30), -1)
        cv2.addWeighted(top_bar, 0.85, frame, 0.15, 0, frame)

        # Title
        cv2.putText(frame, "BIONIC HAND CONTROL", (18, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

        # Connection status badge
        if serial_ctrl.is_connected:
            badge_text = f"SERIAL: {serial_ctrl.port} (115200)"
            badge_color = (0, 200, 100)
        elif serial_ctrl.is_simulated:
            badge_text = "SERIAL: SIMULATION (Mock Mode)"
            badge_color = (0, 190, 255)
        else:
            badge_text = "SERIAL: DISCONNECTED"
            badge_color = (0, 0, 230)

        cv2.rectangle(frame, (380, 12), (730, 43), (35, 40, 50), -1)
        cv2.rectangle(frame, (380, 12), (730, 43), badge_color, 1)
        cv2.circle(frame, (395, 28), 6, badge_color, -1)
        cv2.putText(frame, badge_text, (410, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Arduino HW ACK / Watchdog response badge
        if serial_ctrl.last_ack_response:
            ack_str = serial_ctrl.last_ack_response
            is_warn = "WARN" in ack_str or "ERR" in ack_str
            resp_color = (0, 140, 255) if is_warn else (0, 200, 100)
            cv2.rectangle(frame, (740, 12), (1030, 43), (35, 40, 50), -1)
            cv2.rectangle(frame, (740, 12), (1030, 43), resp_color, 1)
            cv2.putText(frame, f"HW: {ack_str[:22]}", (750, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS & Safety badge
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 240, 255), 2, cv2.LINE_AA)

        # --- Side Telemetry Panel (Finger Gauges) ---
        panel_w = 330
        panel_x = w - panel_w - 15
        panel_y = 65
        panel_h = 320

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (25, 30, 38), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (60, 75, 95), 1)

        cv2.putText(frame, "FINGER TELEMETRY", (panel_x + 15, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 215, 0), 2, cv2.LINE_AA)

        row_y = panel_y + 60
        for finger in config.FINGER_ORDER:
            cfg = config.DEFAULT_FINGER_CONFIGS[finger]
            bend = bends.get(finger, 0.0)
            angle = target_angles.get(finger, cfg.open_angle)
            pct = int(bend * 100)

            # Finger name & angle
            label = f"{cfg.name:<7}: {pct:3d}% -> {angle:3d}°"
            cv2.putText(frame, label, (panel_x + 15, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 235, 245), 1, cv2.LINE_AA)

            # Progress bar background
            bar_x = panel_x + 15
            bar_y = row_y + 7
            bar_w = 295
            bar_h = 10
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 55, 65), -1)

            # Fill bar with color gradient based on flexion
            fill_w = int(bar_w * bend)
            # Green (open) -> Orange/Red (closed)
            r = int(min(255, bend * 2.0 * 255))
            g = int(min(255, (1.0 - bend) * 2.0 * 255))
            bar_color = (40, g, r)
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

            row_y += 50

        # --- Emergency Stop Overlay Banner ---
        if is_estop:
            estop_overlay = frame.copy()
            cv2.rectangle(estop_overlay, (0, h // 2 - 60), (w, h // 2 + 60), (0, 0, 180), -1)
            cv2.addWeighted(estop_overlay, 0.88, frame, 0.12, 0, frame)
            cv2.putText(frame, "EMERGENCY STOP ENGAGED", (w // 2 - 290, h // 2 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "All motion commands halted. (Press SPACE or 'E' to resume)", (w // 2 - 280, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(frame, "NOTICE: Software stop does NOT physically disconnect servo power.", (w // 2 - 270, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 240, 255), 1, cv2.LINE_AA)

        # --- Staged Test Mode Overlay ---
        elif staged_tester.active:
            test_overlay = frame.copy()
            cv2.rectangle(test_overlay, (15, 65), (350, 220), (35, 25, 45), -1)
            cv2.addWeighted(test_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (15, 65), (350, 220), (200, 100, 255), 1)

            cv2.putText(frame, "STAGED HARDWARE TEST", (28, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 140, 255), 2, cv2.LINE_AA)
            sel_f = config.FINGER_ORDER[staged_tester.selected_finger_index]
            sel_cfg = config.DEFAULT_FINGER_CONFIGS[sel_f]
            sel_ang = staged_tester.test_angles[staged_tester.selected_finger_index]

            cv2.putText(frame, f"Active: {sel_cfg.name} (Pin {sel_cfg.pin})", (28, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Angle : {sel_ang} deg [Limit: {sel_cfg.min_safe_angle}-{sel_cfg.max_safe_angle}]", (28, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 120), 1, cv2.LINE_AA)
            cv2.putText(frame, "Keys: [1-5] Select | [UP/DN] Move | [T] Exit", (28, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # --- Lost Tracking Indicator ---
        elif safety_status.startswith("LOST"):
            cv2.rectangle(frame, (18, 68), (340, 110), (20, 20, 90), -1)
            cv2.rectangle(frame, (18, 68), (340, 110), (0, 140, 255), 1)
            cv2.putText(frame, f"HAND LOST: Safety {config.LOST_TRACKING_ACTION} Active", (30, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 120), 2, cv2.LINE_AA)

        # --- Bottom Keyboard Quick Reference Bar ---
        bot_bar = frame.copy()
        cv2.rectangle(bot_bar, (0, h - 42), (w, h), (18, 22, 28), -1)
        cv2.addWeighted(bot_bar, 0.88, frame, 0.12, 0, frame)

        calib_str = "CUSTOM" if calib_mgr.is_custom_calibrated else "DEFAULT"
        shortcut_str = (
            f"[O] Calib Open  [C] Calib Closed  [S] Save  [R] Reset ({calib_str}) | "
            f"[SPACE/E] E-STOP | [T] Stage Test | [Q] Quit"
        )
        cv2.putText(frame, shortcut_str, (18, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 220, 230), 1, cv2.LINE_AA)


# ==============================================================================
# Main Application Loop
# ==============================================================================

def run_application(args):
    """Initializes vision pipeline, serial connection, and runs the real-time loop."""
    print("=================================================================")
    print("      WEBCAM-CONTROLLED FIVE-FINGER BIONIC HAND")
    print("=================================================================")

    # 1. Verify and download MediaPipe model
    model_path = ensure_model_file()

    # 2. Initialize MediaPipe Hand Landmarker
    if not MEDIAPIPE_AVAILABLE:
        print("[!] MediaPipe is not installed. Please run: pip install mediapipe")
        sys.exit(1)

    print("[*] Initializing MediaPipe HandLandmarker ...")
    base_options = BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # 3. Initialize Subsystems
    kinematics = FingerBendCalculator()
    calib_mgr = CalibrationManager()
    angle_mapper = ServoAngleMapper()
    serial_ctrl = SerialHandController(port=args.port, baud=args.baud, simulate=args.simulate)
    safety = SafetyMonitor(serial_ctrl)
    staged_tester = StagedTester(serial_ctrl)
    if args.staged_test:
        staged_tester.active = True

    # 4. Initialize Webcam
    print(f"[*] Opening webcam device index {args.cam} ...")
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[!] ERROR: Could not open camera {args.cam}.")
        print("[!] On macOS, ensure Camera permissions are granted to your terminal/IDE:")
        print("    System Settings -> Privacy & Security -> Camera")
        landmarker.close()
        serial_ctrl.close()
        sys.exit(1)

    print("[+] Camera initialized.")
    print("[+] System running! Position your hand in front of the webcam.")
    print("    Press 'O' with hand open, 'C' with fist closed to calibrate.")
    print("    Press SPACE or 'E' for Emergency Stop.")

    prev_time = time.time()
    frame_count = 0
    fps = 30.0

    current_bends = {f: 0.0 for f in config.FINGER_ORDER}
    current_angles = {f: config.SAFE_REST_ANGLES[f] for f in config.FINGER_ORDER}

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Warning: Failed to read frame from webcam.")
                time.sleep(0.01)
                continue

            frame_count += 1
            now = time.time()
            dt = now - prev_time
            if dt >= 0.5:
                fps = frame_count / dt
                frame_count = 0
                prev_time = now

            # Flip horizontally for natural mirror interaction
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Monotonic millisecond timestamp for VIDEO mode
            timestamp_ms = int(time.time() * 1000)

            # Execute hand landmark detection
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_detected = bool(detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0)
            safety_status = safety.update_hand_presence(hand_detected)

            if hand_detected:
                landmarks = detection_result.hand_landmarks[0]

                # Draw skeleton
                UIOverlay.draw_hand_skeleton(frame, landmarks)

                # Compute kinematics and normalized bends
                raw_metrics = kinematics.compute_raw_metrics(landmarks)
                current_bends = calib_mgr.compute_normalized_bend(raw_metrics)

                # Map to servo angles if not in E-STOP and not in staged test mode
                if not safety.emergency_stop_active and not staged_tester.active:
                    angles, changed = angle_mapper.map_bends_to_angles(current_bends)
                    current_angles = angles

                    # Continuously transmit rate-limited target angles while hand tracking is active.
                    # Rate-limiting to SERIAL_UPDATE_HZ is enforced inside send_angles().
                    # Continuous transmission ensures the physical Arduino watchdog is kept alive.
                    sent = serial_ctrl.send_angles(current_angles)
                    if sent:
                        angle_mapper.commit_angles(current_angles)

            elif safety_status == "LOST_REST":
                current_angles = dict(config.SAFE_REST_ANGLES)
                if not safety.emergency_stop_active and not staged_tester.active:
                    sent = serial_ctrl.send_angles(current_angles)
                    if sent:
                        angle_mapper.commit_angles(current_angles)

            elif safety_status == "LOST_HOLD":
                # In HOLD mode, continue streaming the held target angles rate-limited
                # so the physical Arduino watchdog stays refreshed and maintains position
                if not safety.emergency_stop_active and not staged_tester.active:
                    sent = serial_ctrl.send_angles(current_angles)
                    if sent:
                        angle_mapper.commit_angles(current_angles)

            # If staged single-servo tester is active, keep its target angle refreshed
            if staged_tester.active and not safety.emergency_stop_active:
                staged_tester.update_keepalive()

            # Render UI Overlay
            UIOverlay.draw_hud(
                frame=frame,
                fps=fps,
                safety_status=safety_status,
                is_estop=safety.emergency_stop_active,
                serial_ctrl=serial_ctrl,
                calib_mgr=calib_mgr,
                bends=current_bends,
                target_angles=current_angles,
                staged_tester=staged_tester,
            )

            # Display window
            cv2.imshow("Webcam Five-Finger Bionic Hand", frame)

            # Process Keyboard Controls
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]:  # 'q' or ESC
                print("[*] Quitting application ...")
                break
            elif key in [ord(" "), ord("e"), ord("E")]:  # Emergency Stop toggle
                active = safety.toggle_emergency_stop()
                print(f"[!] EMERGENCY STOP {'ACTIVATED' if active else 'DEACTIVATED'}")
            elif key in [ord("o"), ord("O")]:  # Calibrate Open Hand
                if hand_detected:
                    calib_mgr.record_open(raw_metrics)
                    print("[+] Recorded OPEN HAND calibration reference.")
                else:
                    print("[!] Cannot calibrate: No hand detected.")
            elif key in [ord("c"), ord("C")]:  # Calibrate Closed Fist
                if hand_detected:
                    calib_mgr.record_closed(raw_metrics)
                    print("[+] Recorded CLOSED HAND calibration reference.")
                else:
                    print("[!] Cannot calibrate: No hand detected.")
            elif key in [ord("s"), ord("S")]:  # Save calibration to file
                if calib_mgr.save():
                    print(f"[+] Calibration saved to {config.CALIBRATION_FILE_PATH}")
            elif key in [ord("r"), ord("R")]:  # Reset calibration
                calib_mgr.reset_defaults()
                print("[+] Calibration reset to factory defaults.")
            elif key in [ord("t"), ord("T")]:  # Toggle staged test mode
                staged_tester.active = not staged_tester.active
                print(f"[*] Staged Hardware Test Mode: {'ON' if staged_tester.active else 'OFF'}")
                if not staged_tester.active:
                    serial_ctrl.send_rest()
            elif staged_tester.active:
                # Number keys 1-5 select fingers in staged test
                if key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5")]:
                    idx = key - ord("1")
                    staged_tester.select_finger(idx)
                    print(f"[*] Staged Test: Selected finger {config.DEFAULT_FINGER_CONFIGS[config.FINGER_ORDER[idx]].name}")
                elif key == 0:  # Up arrow in some OpenCV builds
                    staged_tester.step_angle(+5)
                elif key == 1:  # Down arrow in some OpenCV builds
                    staged_tester.step_angle(-5)
                elif key == ord("w") or key == ord("+") or key == ord("="):
                    staged_tester.step_angle(+5)
                elif key == ord("s") or key == ord("-"):
                    staged_tester.step_angle(-5)

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")

    finally:
        print("[*] Performing clean shutdown ...")
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        serial_ctrl.close()
        print("[+] System shutdown complete.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Webcam-Controlled Five-Finger Bionic Hand with MediaPipe and Arduino.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial port path (e.g. /dev/cu.usbmodem101). If omitted, automatically discovers connected Arduino.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=config.SERIAL_BAUD_RATE,
        help="Serial baud rate matching Arduino sketch.",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=config.WEBCAM_INDEX,
        help="Webcam device index (0 for default MacBook FaceTime HD camera).",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode without requiring physical Arduino hardware.",
    )
    parser.add_argument(
        "--staged-test",
        action="store_true",
        help="Start directly in staged single-servo test mode for hardware checkout.",
    )

    args = parser.parse_args()
    run_application(args)


if __name__ == "__main__":
    main()
