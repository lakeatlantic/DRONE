from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
import math

class CartpoleRobot(RobotEmitterReceiverCSV):
	def __init__(self):
		super().__init__()
		self.id = '1'
		timestep = int(self.robot.getBasicTimeStep())
		self.camera = self.robot.getDevice('camera')
		self.camera.enable(timestep)
		self.front_left_led = self.robot.getDevice('front left led')
		self.front_right_led = self.robot.getDevice('front right led')

		self.imu = self.robot.getDevice('inertial unit')
		self.imu.enable(timestep)
		self.gps = self.robot.getDevice('gps')
		self.gps.enable(timestep)
		self.compass = self.robot.getDevice('compass')
		self.compass.enable(timestep)
		self.gyro = self.robot.getDevice('gyro')
		self.gyro.enable(timestep)

		self.camera_roll_motor = self.robot.getDevice('camera roll')
		self.camera_pitch_motor = self.robot.getDevice('camera pitch')

		self.front_left_motor = self.robot.getDevice("front left propeller")
		self.front_right_motor = self.robot.getDevice("front right propeller")
		self.rear_left_motor = self.robot.getDevice("rear left propeller")
		self.rear_right_motor = self.robot.getDevice("rear right propeller")

		self.motors = [self.front_left_motor, self.front_right_motor, self.rear_left_motor, self.rear_right_motor]
		for m in range(4):
			
			self.motors[m].setPosition(math.inf)
			self.motors[m].setVelocity(1.0)

		self.k_vertical_thrust = 68.5
		self.k_vertical_offset = 0.6
		self.k_vertical_p = 3.0
		self.k_roll_p = 50.0
		self.k_pitch_p = 30.0

		self.target_altitude = 0.0
		# print('Hello')
		
	def create_message(self):
		# Read the sensor value, convert to string and save it in a list
		message = self.gps.getValues()
		return message
	
	def SIGN(self, x):
				return ((x) > 0) - ((x) < 0)

	def CLAMP(self, value, low, high): 
		if value < low:
			return low
		else:
			if value > high:
				return high
			else:
				return value

	def handle_receiver(self):
		if self.receiver.getQueueLength() > 0:
			# Receive and decode message from supervisor
			message = self.receiver.getData().decode("utf-8")
			print('>>>>>', message)
			# Convert string message into a list
			message = message.split(",")

			self.use_message_data(message)

			self.receiver.nextPacket()

	def use_message_data(self, message):
		print('message', message)
		action = int(message[0])  # Convert the string message into an action integer
		time = self.robot.getTime()
		roll = self.imu.getRollPitchYaw()[0] + 3.14159 / 2.0
		pitch = self.imu.getRollPitchYaw()[1]
		altitude = self.gps.getValues()[1]
		self.target_altitude = altitude
		roll_acceleration = self.gyro.getValues()[0]
		pitch_acceleration = self.gyro.getValues()[1]

		led_state = (int(time)) % 2

		# self.front_left_led.set(led_state)
		# self.front_right_led.set(led_state)
		
		self.camera_roll_motor.setPosition(-0.115 * roll_acceleration)
		self.camera_pitch_motor.setPosition(-0.1 * pitch_acceleration)
		
		roll_disturbance = 0.0
		pitch_disturbance = 0.0
		yaw_disturbance = 0.0

		if action == 0:
			self.target_altitude += 0.05
		elif action == 1:
			self.target_altitude -= 0.05
		else:
			pass
		
		# Set the motors' velocities based on the action received
		roll_input = self.k_roll_p * self.CLAMP(roll, -1.0, 1.0) + roll_acceleration + roll_disturbance
		pitch_input = self.k_pitch_p * self.CLAMP(pitch, -1.0, 1.0) - pitch_acceleration + pitch_disturbance
		yaw_input = yaw_disturbance
		clamped_difference_altitude = self.CLAMP(self.target_altitude - altitude + self.k_vertical_offset, -1.0, 1.0)
		vertical_input = self.k_vertical_p * pow(clamped_difference_altitude, 3.0)
		
		front_left_motor_input = self.k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
		front_right_motor_input = self.k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
		rear_left_motor_input = self.k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
		rear_right_motor_input = self.k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
		self.front_left_motor.setVelocity(front_left_motor_input)
		self.front_right_motor.setVelocity(-front_right_motor_input)
		self.rear_left_motor.setVelocity(-rear_left_motor_input)
		self.rear_right_motor.setVelocity(rear_right_motor_input)

# Create the robot controller object and run it
robot_controller = CartpoleRobot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it
