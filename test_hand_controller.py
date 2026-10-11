"""Automated Unit and Integration Tests for the Bionic Hand Controller.

Validates:
  - Hardware pin and angle configuration integrity
  - Vector kinematics and finger bend calculation
  - Calibration recording, calculation, and JSON persistence
  - Servo angle mapping, direction reversal, EMA smoothing, and deadband
  - Serial protocol formatting and packet generation
  - Safety monitor timeout and Emergency Stop logic
  - Arduino command parsing rules and edge cases
"""

import json
import math
import os
import tempfile
import unittest

import config
from webcam_hand import (
    CalibrationManager,
    FingerBendCalculator,
    SafetyMonitor,
    SerialHandController,
    ServoAngleMapper,
    StagedTester,
    angle_between_vectors_3d,
    dist_3d,
)


class MockLandmark:
    """Mock for MediaPipe NormalizedLandmark."""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class TestConfiguration(unittest.TestCase):
    """Verifies that configuration matches hardware requirements."""

    def test_finger_order_and_count(self):
        self.assertEqual(len(config.FINGER_ORDER), 5)
        self.assertEqual(config.FINGER_ORDER, ["thumb", "index", "middle", "ring", "little"])

    def test_servo_pin_assignments(self):
        # Thumb: D3, Index: D5, Middle: D6, Ring: D9, Little: D10
        expected_pins = {
            "thumb": 3,
            "index": 5,
            "middle": 6,
            "ring": 9,
            "little": 10,
        }
        for finger, pin in expected_pins.items():
            self.assertIn(finger, config.DEFAULT_FINGER_CONFIGS)
            self.assertEqual(config.DEFAULT_FINGER_CONFIGS[finger].pin, pin)

    def test_angle_limits_safety(self):
        for finger in config.FINGER_ORDER:
            cfg = config.DEFAULT_FINGER_CONFIGS[finger]
            self.assertLess(cfg.min_safe_angle, cfg.max_safe_angle)
            self.assertGreaterEqual(cfg.min_safe_angle, 0)
            self.assertLessEqual(cfg.max_safe_angle, 180)
            # Open and closed angles must stay within safe limits
            self.assertGreaterEqual(cfg.open_angle, cfg.min_safe_angle)
            self.assertLessEqual(cfg.open_angle, cfg.max_safe_angle)
            self.assertGreaterEqual(cfg.closed_angle, cfg.min_safe_angle)
            self.assertLessEqual(cfg.closed_angle, cfg.max_safe_angle)


class TestKinematicsAndBend(unittest.TestCase):
    """Verifies 3D geometry and finger bend calculation."""

    def test_dist_3d(self):
        p1 = (0.0, 0.0, 0.0)
        p2 = (3.0, 4.0, 0.0)
        self.assertAlmostEqual(dist_3d(p1, p2), 5.0)

    def test_angle_between_vectors(self):
        v1 = (1.0, 0.0, 0.0)
        v2 = (0.0, 1.0, 0.0)
        self.assertAlmostEqual(angle_between_vectors_3d(v1, v2), 90.0)

        v_straight1 = (0.0, 1.0, 0.0)
        v_straight2 = (0.0, 1.0, 0.0)
        self.assertAlmostEqual(angle_between_vectors_3d(v_straight1, v_straight2), 0.0)

    def _create_mock_hand_landmarks(self, fingers_extended: bool = True):
        """Generates 21 synthetic hand landmarks representing an extended hand or a fist."""
        lms = [MockLandmark(0.5, 0.8, 0.0)]  # 0: Wrist

        # Thumb: 1: CMC, 2: MCP, 3: IP, 4: TIP
        if fingers_extended:
            lms.append(MockLandmark(0.42, 0.72, 0.0))  # 1: CMC
            lms.append(MockLandmark(0.36, 0.65, 0.0))  # 2: MCP
            lms.append(MockLandmark(0.30, 0.58, 0.0))  # 3: IP
            lms.append(MockLandmark(0.24, 0.52, 0.0))  # 4: TIP (extended away)
        else:
            # Curled toward palm/pinky
            lms.append(MockLandmark(0.44, 0.73, 0.0))
            lms.append(MockLandmark(0.46, 0.66, 0.0))
            lms.append(MockLandmark(0.48, 0.62, 0.0))
            lms.append(MockLandmark(0.52, 0.60, 0.0))  # 4: TIP (folded into palm)

        # 4 Fingers: Index (5-8), Middle (9-12), Ring (13-16), Little (17-20)
        finger_x_bases = [0.42, 0.50, 0.58, 0.65]
        for x_base in finger_x_bases:
            # MCP
            lms.append(MockLandmark(x_base, 0.55, 0.0))
            if fingers_extended:
                # Straight up toward y=0.15
                lms.append(MockLandmark(x_base, 0.42, 0.0))  # PIP
                lms.append(MockLandmark(x_base, 0.28, 0.0))  # DIP
                lms.append(MockLandmark(x_base, 0.15, 0.0))  # TIP
            else:
                # Curled downward back toward MCP (y=0.55)
                lms.append(MockLandmark(x_base, 0.45, -0.05))  # PIP
                lms.append(MockLandmark(x_base, 0.48, 0.05))   # DIP
                lms.append(MockLandmark(x_base, 0.53, 0.02))   # TIP (curled tight)

        return lms

    def test_open_vs_closed_kinematics(self):
        calc = FingerBendCalculator()
        open_lms = self._create_mock_hand_landmarks(fingers_extended=True)
        closed_lms = self._create_mock_hand_landmarks(fingers_extended=False)

        open_metrics = calc.compute_raw_metrics(open_lms)
        closed_metrics = calc.compute_raw_metrics(closed_lms)

        for finger in config.FINGER_ORDER:
            # For all fingers, open posture must produce higher metric than closed posture
            self.assertGreater(
                open_metrics[finger],
                closed_metrics[finger],
                f"Finger {finger} open metric ({open_metrics[finger]}) should be > closed ({closed_metrics[finger]})",
            )

    def test_independent_kinematics_landmark_isolation(self):
        """Proves that flexing any single finger landmark changes only that finger's metric."""
        calc = FingerBendCalculator()
        open_lms = self._create_mock_hand_landmarks(fingers_extended=True)
        base_metrics = calc.compute_raw_metrics(open_lms)

        # Test index finger flex isolation: curl only index finger joints (6, 7, 8)
        index_curled_lms = self._create_mock_hand_landmarks(fingers_extended=True)
        index_curled_lms[6] = MockLandmark(0.42, 0.45, -0.05)  # PIP
        index_curled_lms[7] = MockLandmark(0.42, 0.48, 0.05)   # DIP
        index_curled_lms[8] = MockLandmark(0.42, 0.53, 0.02)   # TIP
        idx_metrics = calc.compute_raw_metrics(index_curled_lms)

        self.assertLess(idx_metrics["index"], base_metrics["index"] - 0.2)
        for other in ["thumb", "middle", "ring", "little"]:
            self.assertAlmostEqual(
                idx_metrics[other], base_metrics[other], places=4,
                msg=f"Finger {other} metric was affected when only index finger moved!"
            )


class TestCalibrationManager(unittest.TestCase):
    """Tests recording, saving, loading, and normalized bend computation."""

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.temp_file.close()

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_recording_and_persistence(self):
        mgr = CalibrationManager(filepath=self.temp_file.name)
        mock_open = {"thumb": 1.4, "index": 0.92, "middle": 0.95, "ring": 0.90, "little": 0.85}
        mock_closed = {"thumb": 0.5, "index": 0.35, "middle": 0.32, "ring": 0.35, "little": 0.35}

        mgr.record_open(mock_open)
        mgr.record_closed(mock_closed)
        self.assertTrue(mgr.save())

        # Load into new manager
        mgr2 = CalibrationManager(filepath=self.temp_file.name)
        self.assertTrue(mgr2.is_custom_calibrated)
        self.assertAlmostEqual(mgr2.open_refs["index"], 0.92)
        self.assertAlmostEqual(mgr2.closed_refs["index"], 0.35)

    def test_normalized_bend_calculation(self):
        mgr = CalibrationManager(filepath=self.temp_file.name)
        mgr.open_refs = {"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "little": 1.0}
        mgr.closed_refs = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0}

        # Open metrics -> bend should be 0.0
        open_metrics = {f: 1.0 for f in config.FINGER_ORDER}
        bends_open = mgr.compute_normalized_bend(open_metrics)
        for f in config.FINGER_ORDER:
            self.assertAlmostEqual(bends_open[f], 0.0)

        # Closed metrics -> bend should be 1.0
        closed_metrics = {f: 0.0 for f in config.FINGER_ORDER}
        bends_closed = mgr.compute_normalized_bend(closed_metrics)
        for f in config.FINGER_ORDER:
            self.assertAlmostEqual(bends_closed[f], 1.0)

        # Halfway metrics -> bend should be 0.5
        half_metrics = {f: 0.5 for f in config.FINGER_ORDER}
        bends_half = mgr.compute_normalized_bend(half_metrics)
        for f in config.FINGER_ORDER:
            self.assertAlmostEqual(bends_half[f], 0.5)

        # Out-of-bounds metrics clamped to [0.0, 1.0]
        oob_metrics = {f: 2.0 for f in config.FINGER_ORDER}
        bends_oob = mgr.compute_normalized_bend(oob_metrics)
        for f in config.FINGER_ORDER:
            self.assertAlmostEqual(bends_oob[f], 0.0)


class TestServoAngleMapper(unittest.TestCase):
    """Tests mapping from normalized bend to servo angles, smoothing, and safety limits."""

    def test_angle_mapping_and_clamping(self):
        mapper = ServoAngleMapper()
        # Full open hand (0.0 bend) with alpha=1.0 (no smoothing lag)
        bends_open = {f: 0.0 for f in config.FINGER_ORDER}
        angles_open, _ = mapper.map_bends_to_angles(bends_open, alpha=1.0)

        for f in config.FINGER_ORDER:
            cfg = config.DEFAULT_FINGER_CONFIGS[f]
            self.assertEqual(angles_open[f], cfg.open_angle)

        # Full closed fist (1.0 bend)
        bends_closed = {f: 1.0 for f in config.FINGER_ORDER}
        angles_closed, _ = mapper.map_bends_to_angles(bends_closed, alpha=1.0)

        for f in config.FINGER_ORDER:
            cfg = config.DEFAULT_FINGER_CONFIGS[f]
            self.assertEqual(angles_closed[f], cfg.closed_angle)

    def test_per_finger_direction_reversal(self):
        # Create configs with index finger reversed
        test_configs = dict(config.DEFAULT_FINGER_CONFIGS)
        test_configs["index"] = config.ServoFingerConfig(
            name="Index",
            pin=5,
            min_safe_angle=10,
            max_safe_angle=170,
            open_angle=20,
            closed_angle=160,
            reversed=True,  # Reversed!
        )
        mapper = ServoAngleMapper(finger_configs=test_configs)

        # At 0.0 bend (open hand), reversed index should go to closed_angle (160)
        bends = {f: 0.0 for f in config.FINGER_ORDER}
        angles, _ = mapper.map_bends_to_angles(bends, alpha=1.0)
        self.assertEqual(angles["index"], 160)

        # At 1.0 bend (closed fist), reversed index should go to open_angle (20)
        bends_closed = {f: 1.0 for f in config.FINGER_ORDER}
        angles_c, _ = mapper.map_bends_to_angles(bends_closed, alpha=1.0)
        self.assertEqual(angles_c["index"], 20)

    def test_ema_smoothing_progression(self):
        mapper = ServoAngleMapper()
        # Set rest initial state (30 deg)
        bends_target = {f: 1.0 for f in config.FINGER_ORDER}  # Target: closed_angle (160)
        alpha = 0.5

        # Step 1: from 30 -> 160 with alpha=0.5 -> expected 30 + 0.5*(160-30) = 95
        angles1, _ = mapper.map_bends_to_angles(bends_target, alpha=alpha)
        for f in config.FINGER_ORDER:
            self.assertEqual(angles1[f], 95)

        # Step 2: from 95 -> 160 with alpha=0.5 -> expected 95 + 0.5*(160-95) = 127.5 -> 128
        angles2, _ = mapper.map_bends_to_angles(bends_target, alpha=alpha)
        for f in config.FINGER_ORDER:
            self.assertEqual(angles2[f], 128)

    def test_independent_finger_bend_mapping_isolation(self):
        """Proves that changing each finger's bend affects ONLY its corresponding servo target."""
        mapper = ServoAngleMapper()
        baseline_bends = {f: 0.0 for f in config.FINGER_ORDER}
        baseline_angles, _ = mapper.map_bends_to_angles(baseline_bends, alpha=1.0)

        for target_finger in config.FINGER_ORDER:
            for test_bend in [0.2, 0.5, 0.8, 1.0]:
                test_bends = dict(baseline_bends)
                test_bends[target_finger] = test_bend

                mapped_angles, _ = mapper.map_bends_to_angles(test_bends, alpha=1.0)

                cfg = config.DEFAULT_FINGER_CONFIGS[target_finger]
                expected_angle = int(round(cfg.open_angle + test_bend * (cfg.closed_angle - cfg.open_angle)))
                self.assertEqual(
                    mapped_angles[target_finger],
                    expected_angle,
                    f"Target finger {target_finger} angle {mapped_angles[target_finger]} != expected {expected_angle}"
                )

                # Ensure ALL OTHER four fingers remain completely unchanged at their baseline angle
                for other_finger in config.FINGER_ORDER:
                    if other_finger != target_finger:
                        self.assertEqual(
                            mapped_angles[other_finger],
                            baseline_angles[other_finger],
                            f"Cross-talk error: Changing {target_finger} altered {other_finger}!"
                        )


class TestSerialProtocolAndSafety(unittest.TestCase):
    """Tests packet formatting, simulation mode, and safety state transitions."""

    def test_packet_formatting_in_simulation(self):
        ctrl = SerialHandController(simulate=True)
        self.assertTrue(ctrl.is_simulated)

        # Packet format test
        test_angles = {"thumb": 30, "index": 45, "middle": 60, "ring": 75, "little": 90}
        success = ctrl.send_angles(test_angles)
        self.assertTrue(success)
        self.assertIn("SIM_ACK:<30,45,60,75,90>", ctrl.last_ack_response)

    def test_emergency_stop_logic(self):
        ctrl = SerialHandController(simulate=True)
        safety = SafetyMonitor(ctrl)

        self.assertFalse(safety.emergency_stop_active)
        # Toggle on
        is_active = safety.toggle_emergency_stop()
        self.assertTrue(is_active)
        self.assertEqual(safety.update_hand_presence(True), "ESTOP")
        self.assertIn("STOP", ctrl.last_ack_response)

        # Toggle off
        is_active = safety.toggle_emergency_stop()
        self.assertFalse(is_active)
        self.assertEqual(safety.update_hand_presence(True), "NORMAL")
        self.assertIn("RESUME", ctrl.last_ack_response)

    def test_loss_of_tracking_timeout(self):
        ctrl = SerialHandController(simulate=True)
        safety = SafetyMonitor(ctrl)

        # Hand detected initially
        status = safety.update_hand_presence(True)
        self.assertEqual(status, "NORMAL")

        # Simulate time passing with hand lost
        safety.last_seen_hand_time = safety.last_seen_hand_time - (config.LOST_TRACKING_TIMEOUT_SEC + 0.1)
        status_lost = safety.update_hand_presence(False)
        self.assertTrue(status_lost.startswith("LOST"))

    def test_continuous_rate_limiting_and_transmission(self):
        """Verifies that send_angles correctly rate-limits continuous calls to SERIAL_UPDATE_HZ."""
        ctrl = SerialHandController(simulate=True)
        angles = {"thumb": 30, "index": 45, "middle": 60, "ring": 75, "little": 90}

        # 1. First call transmits successfully
        self.assertTrue(ctrl.send_angles(angles))

        # 2. Immediate second call is suppressed by rate limiter
        self.assertFalse(ctrl.send_angles(angles))

        # 3. Advancing time past SERIAL_UPDATE_INTERVAL enables next transmission
        ctrl.last_tx_time -= (config.SERIAL_UPDATE_INTERVAL + 0.005)
        self.assertTrue(ctrl.send_angles(angles))

    def test_staged_tester_commands_and_keepalive(self):
        """Verifies that StagedTester sends framed TEST commands and refreshes keepalive."""
        ctrl = SerialHandController(simulate=True)
        tester = StagedTester(ctrl)
        tester.active = True
        tester.select_finger(1)  # Index finger

        tester.step_angle(+10)  # 30 + 10 = 40
        self.assertEqual(tester.test_angles[1], 40)
        self.assertIn("SIM_ACK:<TEST,1,40>", ctrl.last_ack_response)

        # Keepalive refresh after interval
        tester.last_tx_time -= 0.6
        tester.update_keepalive()
        self.assertIn("SIM_ACK:<TEST,1,40>", ctrl.last_ack_response)


class TestArduinoProtocolParsing(unittest.TestCase):
    """Simulates the Arduino C++ parser logic to ensure protocol compatibility."""

    @staticmethod
    def simulate_arduino_parse(packet_str: str, min_angles=(10, 10, 10, 10, 10), max_angles=(170, 170, 170, 170, 170)):
        """Python model of packet parsing in arduino/servo_controller.ino."""
        packet_str = packet_str.strip()
        if packet_str.startswith("<") and packet_str.endswith(">"):
            packet_str = packet_str[1:-1].strip()

        if packet_str in ["STOP", "ESTOP"]:
            return "ACK:STOPPED"
        if packet_str == "RESUME":
            return "ACK:RESUMED"
        if packet_str == "REST":
            return "ACK:REST"
        if packet_str == "PING":
            return "PONG"
        if packet_str.startswith("TEST,"):
            parts = packet_str.split(",")
            if len(parts) != 3:
                return "ERR:TEST_FORMAT"
            idx, ang = int(parts[1]), int(parts[2])
            if not (0 <= idx < 5):
                return "ERR:TEST_INVALID_INDEX"
            clamped = max(min_angles[idx], min(max_angles[idx], ang))
            return f"ACK:TEST,{idx},{clamped}"

        # Comma-separated angles
        tokens = packet_str.split(",")
        if len(tokens) != 5:
            return f"ERR:EXPECTED_5_ANGLES_GOT_{len(tokens)}"

        clamped_angles = []
        for i, tok in enumerate(tokens):
            tok = tok.strip()
            if not tok:
                return "ERR:EMPTY_TOKEN"
            try:
                val = int(tok)
            except ValueError:
                return "ERR:INVALID_INT"
            clamped = max(min_angles[i], min(max_angles[i], val))
            clamped_angles.append(clamped)

        return "ACK:" + ",".join(str(a) for a in clamped_angles)

    def test_arduino_parsing_valid(self):
        res = self.simulate_arduino_parse("30,45,60,75,90")
        self.assertEqual(res, "ACK:30,45,60,75,90")

    def test_arduino_parsing_framed_packets(self):
        res = self.simulate_arduino_parse("<30,45,60,75,90>\n")
        self.assertEqual(res, "ACK:30,45,60,75,90")
        res_test = self.simulate_arduino_parse("<TEST,3,70>\n")
        self.assertEqual(res_test, "ACK:TEST,3,70")
        res_rest = self.simulate_arduino_parse("<REST>\n")
        self.assertEqual(res_rest, "ACK:REST")

    def test_arduino_parsing_clamping(self):
        # Out-of-bounds requested angles clamped between 10 and 170
        res = self.simulate_arduino_parse("0,200,5,180,90")
        self.assertEqual(res, "ACK:10,170,10,170,90")

    def test_arduino_parsing_commands(self):
        self.assertEqual(self.simulate_arduino_parse("STOP"), "ACK:STOPPED")
        self.assertEqual(self.simulate_arduino_parse("RESUME"), "ACK:RESUMED")
        self.assertEqual(self.simulate_arduino_parse("REST"), "ACK:REST")
        self.assertEqual(self.simulate_arduino_parse("PING"), "PONG")
        self.assertEqual(self.simulate_arduino_parse("TEST,2,50"), "ACK:TEST,2,50")

    def test_arduino_parsing_invalid(self):
        self.assertTrue(self.simulate_arduino_parse("10,20,30").startswith("ERR:EXPECTED_5_ANGLES"))
        self.assertEqual(self.simulate_arduino_parse("10,20,invalid,40,50"), "ERR:INVALID_INT")


if __name__ == "__main__":
    unittest.main()
