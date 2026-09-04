import subprocess
import sys
import time
from pathlib import Path

TONYPI_DIR = Path("/home/pi/TonyPi")
HIWONDER_SDK_DIR = TONYPI_DIR / "HiwonderSDK"
ACTION_GROUP_DIR = TONYPI_DIR / "ActionGroups"
for extra_path in (TONYPI_DIR, HIWONDER_SDK_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

AGC = None
rrc = None
Controller = None
board = None
ctl = None


def _ensure_hardware():
    global AGC, rrc, Controller, board, ctl
    if AGC is not None and ctl is not None:
        return

    try:
        import hiwonder.ActionGroupControl as action_group_control  # 动作库，必须包含该库
        import hiwonder.ros_robot_controller_sdk as ros_sdk
        from hiwonder.Controller import Controller as controller_cls
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The TonyPi hardware SDK is not available. Install the Hiwonder runtime dependencies before using motion helpers."
        ) from exc

    AGC = action_group_control
    rrc = ros_sdk
    Controller = controller_cls
    board = rrc.Board()
    ctl = Controller(board)


def turn_right(tt):
    """向右转头，可转0~60度"""
    _ensure_hardware()
    assert 0 <= tt <= 60, "angle must be in range [0, 60]"
    ctl.set_pwm_servo_pulse(2, 1500 - tt * 10, 1500)
    time.sleep(2)


def turn_left(tt):
    """向左转头，可转0~60度"""
    _ensure_hardware()
    assert 0 <= tt <= 60, "angle must be in range [0, 60]"
    ctl.set_pwm_servo_pulse(2, 1500 + tt * 10, 1500)
    time.sleep(2)


def turn_ahead():
    """转头回正"""
    _ensure_hardware()
    ctl.set_pwm_servo_pulse(2, 1500, 1500)
    time.sleep(2)


def raise_head(tt):
    """抬头，可抬0~60度"""
    _ensure_hardware()
    assert 0 <= tt <= 60, "angle must be in range [0, 60]"
    ctl.set_pwm_servo_pulse(1, 1500 + tt * 10, 1500)
    time.sleep(2)


def lower_head(tt):
    """低头，可低0~60度"""
    _ensure_hardware()
    assert 0 <= tt <= 60, "angle must be in range [0, 60]"
    ctl.set_pwm_servo_pulse(1, 1500 - tt * 10, 1500)
    time.sleep(2)


def reset_head():
    """上下复位"""
    _ensure_hardware()
    ctl.set_pwm_servo_pulse(1, 1500, 1500)
    time.sleep(2)


def capture_image(filename="noname"):
    """拍照"""
    timestamp = int(time.time())
    if filename == "noname":
        filename = f"/home/pi/Pictures/photo_{timestamp}.jpg"
    cmd = f"fswebcam -r 2592x1944 --no-banner -S 3 {filename}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"拍照失败: {result.stderr}")
        return None

    print(f"照片已保存: {filename}")
    return filename


def act(action_name, repeat_time=1, stand=True):
    """执行动作组
    action list for example:
    stand
    go_forward , go_forward_fast , go_forward_one_step , go_forward_one_small_step
    back , back_fast , back_one_step
    turn_left , turn_right , turn_left_small_step , turn_right_small_step , turn_left_fast , turn_right_fast
    lift_left_hand , seize_right
    """
    _ensure_hardware()
    AGC.runActionGroup(action_name, times=repeat_time, with_stand=stand, path=str(ACTION_GROUP_DIR) + "/")
