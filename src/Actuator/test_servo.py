import time
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c, address=0x40)
pca.frequency = 50

print("Test balayage duty cycle sur channel 0")
for duty in range(1500, 8000, 200):
    pca.channels[0].duty_cycle = duty
    print(duty)
    time.sleep(0.15)

pca.channels[0].duty_cycle = 0