"""controller_py controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
import math

def SIGN(x):
    return ((x) > 0) - ((x) < 0)
def CLAMP(value, low, high): 
    if value < low:
        return low
    else:
        if value > high:
            return high
        else:
            return value
    # return ((value) < (low) ? (low) : ((value) > (high) ? (high) : (value)))

# create the Robot instance.
robot = Robot()
keyboard = Keyboard()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

camera = robot.getDevice('camera')
print('camera', camera)
camera.enable(timestep)
front_left_led = robot.getDevice('front left led')
print('front left led', front_left_led)
front_right_led = robot.getDevice('front right led')
print('front right led', front_right_led)
imu = robot.getDevice('inertial unit')
print('imu', imu)
imu.enable(timestep)
gps = robot.getDevice('gps')
print('gps', gps)
gps.enable(timestep)
compass = robot.getDevice('compass')
print('compass', compass)
compass.enable(timestep)
gyro = robot.getDevice('gyro')
print('gyro', gyro)
gyro.enable(timestep)
keyboard.enable(timestep)
camera_roll_motor = robot.getDevice('camera roll')
print('camera_roll_motor', camera_roll_motor)
camera_pitch_motor = robot.getDevice('camera pitch')
print('camera_pitch_motor', camera_pitch_motor)

front_left_motor = robot.getDevice("front left propeller")
print('front_left_motor', front_left_motor)
front_right_motor = robot.getDevice("front right propeller")
print('front_right_motor', front_right_motor)
rear_left_motor = robot.getDevice("rear left propeller")
print('rear_left_motor', rear_left_motor)
rear_right_motor = robot.getDevice("rear right propeller")
print('rear_right_motor', rear_right_motor)

motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]
for m in range(4):
    
    motors[m].setPosition(math.inf)
    motors[m].setVelocity(1.0)

print("Start the drone...\n")
print("You can control the drone with your computer keyboard:\n")
print("- 'up': move forward.\n")
print("- 'down': move backward.\n")
print("- 'right': turn right.\n")
print("- 'left': turn left.\n")
print("- 'shift + up': increase the target altitude.\n")
print("- 'shift + down': decrease the target altitude.\n")
print("- 'shift + right': strafe right.\n")
print("- 'shift + left': strafe left.\n")

k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 3.0
k_roll_p = 50.0
k_pitch_p = 30.0

target_altitude = 0

while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    
    time = robot.getTime()
    print('time', time)
    roll = imu.getRollPitchYaw()[0] + 3.14159 / 2.0
    print('roll', roll)
    pitch = imu.getRollPitchYaw()[1]
    print('pitch', pitch)
    altitude = gps.getValues()[1]
    print('altitude', altitude)
    roll_acceleration = gyro.getValues()[0]
    print('roll_acceleration', roll_acceleration)
    pitch_acceleration = gyro.getValues()[1]
    print('pitch_acceleration', pitch_acceleration)
    
    led_state = (int(time)) % 2

    front_left_led.set(led_state)
    front_right_led.set(led_state)
    
    camera_roll_motor.setPosition(-0.115 * roll_acceleration)
    camera_pitch_motor.setPosition(-0.1 * pitch_acceleration)
    
    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0
    key = keyboard.getKey()
    while key > 0:
        print('key',key)
        if key == 315: # WB_KEYBOARD_UP
            pitch_disturbance = 2.0
            break
        elif key == 317: # WB_KEYBOARD_DOWN
          pitch_disturbance = -2.0
          break
        elif key == 316: # WB_KEYBOARD_RIGHT
          yaw_disturbance = 1.3
          break
        elif key == 314: # WB_KEYBOARD_LEFT
          yaw_disturbance = -1.3
          break
        elif key == 65852: # WB_KEYBOARD_SHIFT + WB_KEYBOARD_RIGHT
          roll_disturbance = -1.0
          break
        elif key == 65850: # WB_KEYBOARD_SHIFT + WB_KEYBOARD_LEFT
          roll_disturbance = 1.0
          break
        elif key == 65851: # WB_KEYBOARD_SHIFT + WB_KEYBOARD_UP
          target_altitude += 0.05
          print("target altitude: %", target_altitude, " [m]\n")
          break;
        elif key == 65853: # WB_KEYBOARD_SHIFT + WB_KEYBOARD_DOWN
          target_altitude -= 0.05
          print("target altitude: %", target_altitude, " [m]\n")
          break
        key = keyboard.getKey()
     
    print('k_roll_p', k_roll_p)   
    roll_input = k_roll_p * CLAMP(roll, -1.0, 1.0) + roll_acceleration + roll_disturbance
    pitch_input = k_pitch_p * CLAMP(pitch, -1.0, 1.0) - pitch_acceleration + pitch_disturbance
    yaw_input = yaw_disturbance
    clamped_difference_altitude = CLAMP(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
    vertical_input = k_vertical_p * pow(clamped_difference_altitude, 3.0)
    
    front_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
    rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    t = gps.getValues()
    print(t)
    front_left_motor.setVelocity(front_left_motor_input)
    front_right_motor.setVelocity(-front_right_motor_input)
    rear_left_motor.setVelocity(-rear_left_motor_input)
    rear_right_motor.setVelocity(rear_right_motor_input)

robot.cleanup()
# Enter here exit cleanup code.
