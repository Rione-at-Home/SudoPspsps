from .dynamixel_driver import DynamixelDriver
import time

driver = DynamixelDriver()

# Verify communication first
driver.ping(1)
driver.ping(2)

# Store current positions as zero
driver.calibrate_zero()

driver.enable()

print("Center")
driver.set_pan(0)
driver.set_tilt(0)
time.sleep(2)

print("Right")
driver.set_pan(30)
time.sleep(2)

print("Center")
driver.set_pan(0)
time.sleep(2)

print("Left")
driver.set_pan(-30)
time.sleep(2)

print("Center")
driver.set_pan(0)
time.sleep(2)

print("Tilt Up")
driver.set_tilt(10)
time.sleep(2)

print("Center")
driver.set_tilt(0)
time.sleep(2)

driver.disable()
driver.close()