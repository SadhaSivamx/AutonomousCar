
import RPi.GPIO as GPIO
import time

#Pin Definitions
in1 = 24
in2 = 23
en = 25
in3 = 16
in4 = 20
enx = 21

#OutputMode GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)

GPIO.setup(en, GPIO.OUT)
GPIO.setup(enx, GPIO.OUT)

#Setting to 0
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)

#Speed Setting
p = GPIO.PWM(en, 1000)
p.start(50)
q = GPIO.PWM(enx, 1000)
q.start(50)

print("\n")
def init():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(in1, GPIO.OUT)
    GPIO.setup(in2, GPIO.OUT)
    GPIO.setup(in3, GPIO.OUT)
    GPIO.setup(in4, GPIO.OUT)
    GPIO.setup(en, GPIO.OUT)
    GPIO.setup(enx, GPIO.OUT)


def forward():
    print("Motion Forward")
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)


def backward():
    print("Motion Backward")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)

def left():
    print("Motion Left")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)

def right():
    print("Motion Right")
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)
def stop():
    print("Motion NULL")
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)

#Before Starting the Motor's
GPIO.cleanup()
init()

#Body
timex=1
forward()
time.sleep(timex)
backward()
time.sleep(timex)
left()
time.sleep(timex)
right()
time.sleep(timex)
stop()
#After Ending Stuff
GPIO.cleanup()
