"""
ADAS ECU Actuator Controller & Live ASCII Cockpit Dashboard Simulation
Translates visual perception telemetry (distance, TTC, lane offset, signals)
into real-time simulated CAN-Bus vehicle actuator commands (Throttle, Brake,
Electric Power Steering - EPS, Cabin Warnings).
"""
import sys
import time
import os
import shutil

# Ensure Windows stdout supports UTF-8 characters if available
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class ADASActuatorSimulator:
    """
    Simulates vehicle low-level ECU actuator controllers:
    - Electronic Throttle Control (ETC)
    - Hydraulic Brake Assist / Autonomous Emergency Braking (AEB)
    - Electric Power Steering (EPS / Lane Keep Assist)
    - Driver Alerts (Audio Buzzer, Visual HUD, Hazard Flashers)
    """

    def __init__(self, cruise_throttle_pct=60.0):
        self.cruise_throttle_pct = cruise_throttle_pct

    def compute_actuator_commands(
        self,
        critical_state: str,
        closest_dist: float,
        target_speed_kmh: float,
        ttc_s: float,
        offset_m: float,
        ldw_state: str,
        traffic_signal: str,
    ):
        """
        Takes perception inputs and returns a structured dictionary of actuator commands.
        """
        # 1. Longitudinal Control: Throttle & Brake
        throttle_pct = self.cruise_throttle_pct
        brake_pct = 0.0
        decel_target_ms2 = 0.0
        throttle_mode = "CRUISE_ACTIVE"
        brake_mode = "STANDBY"
        hazard_flashers = False
        audio_chime = "SILENT"

        if critical_state == "AEB_BRAKE":
            throttle_pct = 0.0
            throttle_mode = "OVERRIDE_CUT"
            brake_pct = 100.0
            brake_mode = "EMERGENCY_AEB_MAX"
            decel_target_ms2 = -9.2
            hazard_flashers = True
            audio_chime = "CRITICAL_CONTINUOUS_ALARM"
        elif critical_state == "FCW_WARN":
            throttle_pct = 0.0
            throttle_mode = "OVERRIDE_CUT"
            brake_pct = 40.0
            brake_mode = "PRE_FILL_HYDRAULIC"
            decel_target_ms2 = -3.5
            audio_chime = "PULSED_FCW_WARNING"
        elif "RED" in traffic_signal:
            throttle_pct = 0.0
            throttle_mode = "SIGNAL_OVERRIDE"
            brake_pct = 55.0
            brake_mode = "SERVICE_BRAKE_STOP"
            decel_target_ms2 = -4.0
        elif "YELLOW" in traffic_signal:
            throttle_pct = 20.0
            throttle_mode = "COAST_CAUTION"
            brake_pct = 15.0
            brake_mode = "PRE_BRAKE"
            decel_target_ms2 = -1.5
        else:
            # Safe driving state
            throttle_pct = self.cruise_throttle_pct
            brake_pct = 0.0
            brake_mode = "DISENGAGED"
            decel_target_ms2 = 0.0

        # 2. Lateral Control: Electric Power Steering (EPS / LKA)
        eps_torque_nm = 0.0
        steer_angle_deg = 0.0
        eps_mode = "LKA_PASSIVE"

        if ldw_state == "DRIFT_RIGHT":
            eps_torque_nm = float(max(-3.5, -2.0 - (offset_m * 1.5)))
            steer_angle_deg = float(max(-6.0, -offset_m * 8.0))
            eps_mode = "CORRECT_LEFT [<<<]"
            if audio_chime == "SILENT":
                audio_chime = "LANE_DEPARTURE_BEEP"
        elif ldw_state == "DRIFT_LEFT":
            eps_torque_nm = float(min(3.5, 2.0 + (abs(offset_m) * 1.5)))
            steer_angle_deg = float(min(6.0, abs(offset_m) * 8.0))
            eps_mode = "CORRECT_RIGHT [>>>]"
            if audio_chime == "SILENT":
                audio_chime = "LANE_DEPARTURE_BEEP"
        else:
            eps_torque_nm = 0.0
            steer_angle_deg = 0.0
            eps_mode = "LANE_CENTERED [OK]"

        return {
            "throttle_pct": throttle_pct,
            "throttle_mode": throttle_mode,
            "brake_pct": brake_pct,
            "brake_mode": brake_mode,
            "decel_target_ms2": decel_target_ms2,
            "eps_torque_nm": eps_torque_nm,
            "steer_angle_deg": steer_angle_deg,
            "eps_mode": eps_mode,
            "hazard_flashers": hazard_flashers,
            "audio_chime": audio_chime,
        }


class LiveTerminalDashboard:
    """
    Renders an active ASCII / ANSI instrument cluster dashboard in the terminal.
    """

    # ANSI Color Codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"
    BG_GREEN = "\033[42m"

    def __init__(self, use_colors=True, max_log_history=3):
        self.use_colors = use_colors
        self.max_log_history = max_log_history
        self.log_history = []
        self.simulator = ADASActuatorSimulator()

    def _color(self, text, code):
        if not self.use_colors:
            return text
        return f"{code}{text}{self.RESET}"

    def _make_bar(self, pct, length=18, color="\033[92m"):
        pct = max(0.0, min(100.0, pct))
        filled_len = int(round(length * pct / 100))
        empty_len = length - filled_len
        bar = ("#" * filled_len) + ("." * empty_len)
        if self.use_colors:
            return f"{color}{bar}{self.RESET}"
        return bar

    def update_and_render(
        self,
        frame_idx: int,
        total_frames: int,
        fps: float,
        critical_state: str,
        closest_dist: float,
        primary_target_info: dict,
        offset_m: float,
        ldw_state: str,
        traffic_signal: str,
        detection_counts: dict,
    ):
        """
        Renders the active dashboard frame to stdout.
        """
        t_sec = frame_idx / max(fps, 1.0)
        minutes = int(t_sec // 60)
        seconds = t_sec % 60

        target_speed = primary_target_info.get("speed_kmh", 0.0)
        target_ttc = primary_target_info.get("ttc_s", float("inf"))
        target_class = primary_target_info.get("class_name", "None")
        target_id = primary_target_info.get("track_id", "--")

        actuators = self.simulator.compute_actuator_commands(
            critical_state=critical_state,
            closest_dist=closest_dist,
            target_speed_kmh=target_speed,
            ttc_s=target_ttc,
            offset_m=offset_m,
            ldw_state=ldw_state,
            traffic_signal=traffic_signal,
        )

        # Record significant ECU override events into log history
        if critical_state in ["AEB_BRAKE", "FCW_WARN"] or "RED" in traffic_signal or ldw_state != "CENTERED":
            event_tag = (
                f"[{minutes:02d}:{seconds:04.1f}s] F#{frame_idx:04d} -> "
                f"SYS:{critical_state} | BRAKE:{actuators['brake_pct']:.0f}% | "
                f"EPS:{actuators['eps_torque_nm']:+.1f}Nm | SIGNAL:{traffic_signal}"
            )
            if not self.log_history or self.log_history[-1] != event_tag:
                self.log_history.append(event_tag)
                if len(self.log_history) > self.max_log_history:
                    self.log_history.pop(0)

        # Format Master Status Badge
        if critical_state == "AEB_BRAKE":
            status_badge = self._color("[! EMERGENCY BRAKE (AEB) !]", f"{self.BOLD}{self.RED}")
        elif critical_state == "FCW_WARN":
            status_badge = self._color("[! FORWARD COLLISION WARN !]", f"{self.BOLD}{self.YELLOW}")
        elif "RED" in traffic_signal:
            status_badge = self._color("[ TRAFFIC LIGHT STOP ]", f"{self.BOLD}{self.RED}")
        elif ldw_state != "CENTERED":
            status_badge = self._color(f"[ LKA: {ldw_state} ]", f"{self.BOLD}{self.YELLOW}")
        else:
            status_badge = self._color("[ CRUISE - ALL CLEAR ]", f"{self.BOLD}{self.GREEN}")

        # Format Target String
        if closest_dist < 120 and target_class != "None":
            ttc_str = f"{target_ttc:.1f}s" if target_ttc < 30 else "N/A"
            target_str = (
                f"#{target_id} {target_class.upper()} @ {closest_dist:.1f}m | "
                f"RelVel: {target_speed:+.0f}km/h | TTC: {ttc_str}"
            )
        else:
            target_str = "No hazardous targets in path"

        # Format Lane String
        if ldw_state == "CENTERED":
            lane_str = self._color(f"Centered ({offset_m:+.2f}m)", self.GREEN)
        elif ldw_state == "DRIFT_RIGHT":
            lane_str = self._color(f"Drifting Right ({offset_m:+.2f}m) --->", self.YELLOW)
        else:
            lane_str = self._color(f"<--- Drifting Left ({offset_m:+.2f}m)", self.YELLOW)

        # Format Signal String
        if "RED" in traffic_signal:
            sig_str = self._color("RED (STOP)", f"{self.BOLD}{self.RED}")
        elif "YELLOW" in traffic_signal:
            sig_str = self._color("YELLOW (CAUTION)", f"{self.BOLD}{self.YELLOW}")
        elif "GREEN" in traffic_signal:
            sig_str = self._color("GREEN (GO)", f"{self.BOLD}{self.GREEN}")
        else:
            sig_str = self._color("CLEAR / NONE", self.DIM)

        # Format Actuator Bars
        throttle_color = self.GREEN if actuators["throttle_pct"] > 0 else self.DIM
        brake_color = self.RED if actuators["brake_pct"] > 50 else (self.YELLOW if actuators["brake_pct"] > 0 else self.DIM)

        thr_bar = self._make_bar(actuators["throttle_pct"], length=18, color=throttle_color)
        brk_bar = self._make_bar(actuators["brake_pct"], length=18, color=brake_color)
        steer_bar = self._format_steering_gauge(actuators["eps_torque_nm"])

        # Construct clean ASCII lines (ASCII-safe border for Windows compatibility)
        lines = []
        w_box = 78
        sep_double = "=" * w_box
        sep_single = "-" * w_box

        lines.append(f"+{sep_double}+")
        title = self._color("ADAS LEVEL 2 ECU & ACTUATOR TELEMETRY DASHBOARD", self.BOLD + self.CYAN)
        lines.append(f"|  {title}  |")
        lines.append(f"+{sep_double}+")
        
        progress_str = f"Frame: {frame_idx:05d}/{total_frames:05d} | Time: {minutes:02d}:{seconds:04.1f}s | FPS: {fps:.1f}"
        lines.append(f"| {progress_str} | Status: {status_badge}")
        lines.append(f"+{sep_single}+")

        lines.append(f"| [ PERCEPTION TELEMETRY ]")
        lines.append(f"|   * Lead Target    : {target_str}")
        lines.append(f"|   * Lane Position  : {lane_str}")
        lines.append(f"|   * Traffic Light  : {sig_str}")
        
        counts_str = ", ".join([f"{v} {k}" for k, v in detection_counts.items() if v > 0]) or "None"
        lines.append(f"|   * Active Trackers: {counts_str}")
        lines.append(f"+{sep_single}+")

        lines.append(f"| [ ACTUATOR CONTROLLER OUTPUTS (CAN-BUS SIMULATION) ]")
        lines.append(f"|   * Throttle Cmd   : [{thr_bar}] {actuators['throttle_pct']:5.1f}% | Mode: {actuators['throttle_mode']}")
        lines.append(f"|   * Brake Pressure : [{brk_bar}] {actuators['brake_pct']:5.1f}% | Decel: {actuators['decel_target_ms2']:+.1f} m/s2 ({actuators['brake_mode']})")
        lines.append(f"|   * EPS Steering   : {steer_bar} | Torque: {actuators['eps_torque_nm']:+4.1f} Nm | Angle: {actuators['steer_angle_deg']:+4.1f} deg")
        
        alarm_str = f"Audio: {actuators['audio_chime']} | Hazards: {'ON (FLASHING)' if actuators['hazard_flashers'] else 'OFF'}"
        alarm_color = self.RED if actuators["hazard_flashers"] or "ALARM" in actuators["audio_chime"] else self.DIM
        lines.append(f"|   * Cabin Alerts   : {self._color(alarm_str, alarm_color)}")

        if self.log_history:
            lines.append(f"+{sep_single}+")
            lines.append(f"| [ RECENT ECU DISPATCH LOGS ]")
            for entry in self.log_history[-3:]:
                lines.append(f"|   {self._color(entry, self.DIM)}")

        lines.append(f"+{sep_double}+")

        output_block = "\n".join(lines)
        try:
            sys.stdout.write("\033[H\033[J" + output_block + "\n")
            sys.stdout.flush()
        except Exception:
            # Fallback for plain environments
            print(output_block)

    def _format_steering_gauge(self, torque_nm):
        clamped = max(-3.5, min(3.5, torque_nm))
        gauge = list(".....|.....")
        center = 5
        if clamped < -0.3:
            num_arrows = int(round(abs(clamped) / 3.5 * 5))
            for i in range(center - num_arrows, center):
                gauge[i] = "<"
        elif clamped > 0.3:
            num_arrows = int(round(clamped / 3.5 * 5))
            for i in range(center + 1, center + 1 + num_arrows):
                gauge[i] = ">"
        gauge_str = "".join(gauge)
        color = self.YELLOW if abs(torque_nm) > 0.5 else self.GREEN
        return f"[{self._color(gauge_str, color)}]"
