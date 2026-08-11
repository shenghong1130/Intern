import hiwonder.ActionGroupControl as AGC
import time

# 把你想测试的动作放进去
test_list = ['put_down_object']

for action in test_list:
    print(f"正在测试动作: {action}")
    AGC.runActionGroup('stand', times=1)
    time.sleep(1)
    AGC.runActionGroup(action, times=1)
    time.sleep(2) # 给足时间看高度
    AGC.runActionGroup('stand', times=1)
    time.sleep(1)