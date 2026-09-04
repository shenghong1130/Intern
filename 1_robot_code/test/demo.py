import time
import hiwonder.ActionGroupControl as AGC    #动作库，必须包含该库
import subprocess   #拍照

import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller
board = rrc.Board()
ctl = Controller(board)     #转头准备

def capture_image():
    """拍照"""
    timestamp = int(time.time())
    filename = f"/home/pi/Pictures/photo_{timestamp}.jpg"
    cmd = f"fswebcam -r 2592x1944 --no-banner -S 3 {filename}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"拍照失败: {result.stderr}")
        return None
    
    print(f"照片已保存: {filename}")
    return filename

def turn_right_30():
    ctl.set_pwm_servo_pulse(2, 1200, 1000)
    time.sleep(2)

def turn_left_30():
    ctl.set_pwm_servo_pulse(2, 1800, 1000)
    time.sleep(2)

def turn_ahead():
    ctl.set_pwm_servo_pulse(2, 1500, 1000)
    time.sleep(2)
# 说明：函数内第一个参数对应转头角度，1500为正，按1:10偏离，增加为左。可以自定义其他角度的类似函数

AGC.runActionGroup('stand')                                                        # 参数为动作组的名称，不包含后缀，以字符形式传入(the parameter is the name of the action group, without the file extension, passed as a string)
AGC.runActionGroup('go_forward_fast', times=10, with_stand=True)                         # 第二个参数为运行动作次数，默认1, 当为0时表示循环运行， 第三个参数表示最后是否以立正姿态收步(the second parameter is the number of times the action should run, defaulting to 1. When set to 0, it indicates continuous looping. The third parameter indicates whether to end in the 'stand' posture after the final action)

capture_image()
turn_left_30()
capture_image()
turn_right_30()
capture_image()
turn_ahead()


AGC.runActionGroup('turn_right',times=5)#小步转弯

AGC.runActionGroup('go_forward_fast', times=10, with_stand=True)    
AGC.runActionGroup('stand')