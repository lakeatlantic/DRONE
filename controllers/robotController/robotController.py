from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
import math
from abc import abstractmethod
from collections.abc import Iterable
import numpy as np

class CartpoleRobot(RobotEmitterReceiverCSV):
	def __init__(self):
		super().__init__()
		self.name = self.robot.getName()
		timestep = int(self.robot.getBasicTimeStep())
		self.camera = self.robot.getDevice('camera')
		# self.camera.enable(timestep)
		self.camera.disable()
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

		# self.k_vertical_thrust = 102.75 #68.5
		# self.k_vertical_offset = 0.9 #0.6
		# self.k_vertical_p = 4.5 #3.0
		# self.k_roll_p = 75.0 #50.0
		# self.k_pitch_p = 45.0 #30.0

		self.k_vertical_thrust = 68.5
		self.k_vertical_offset = 0.6
		self.k_vertical_p = 3.0
		self.k_roll_p = 50.0
		self.k_pitch_p = 30.0

		# self.k_vertical_thrust = 68.5
		# self.k_vertical_offset = 0.6
		# self.k_vertical_p = 3.0
		# self.k_roll_p = 50.0
		# self.k_pitch_p = 30.0

		self.target_altitude = 1.0
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

	def initialize_comms(self, emitter_name, receiver_name):
		emitter = self.robot.getDevice("emitter")
		receiver = self.robot.getDevice("receiver")
		receiver.enable(self.timestep)
		return emitter, receiver
		
	def handle_emitter(self):
		data = self.create_message()

		assert isinstance(data,
						  Iterable), "The action object should be Iterable"

		string_message = ""
		# message can either be a list that needs to be converted in a string
		# or a straight-up string
		if type(data) is list:
			string_message = ",".join(map(str, data))
		elif type(data) is str:
			string_message = data
		else:
			raise TypeError(
				"message must be either a comma-separated string or a 1D list")

		string_message = string_message.encode("utf-8")
		# self.emitter.send(string_message)

	def handle_receiver(self):
		if self.receiver.getQueueLength() > 0:
			# Receive and decode message from supervisor
			message = self.receiver.getData().decode("utf-8")
			# Convert string message into a list
			message = message.split(",")

			self.use_message_data(message)

			self.receiver.nextPacket()

	def use_message_data(self, message):
		cam_roll = self.robot.getDevice("camera roll").getTargetPosition()
		if cam_roll < -0.5 or cam_roll > 0.5:
			self.robot.setCustomData("unstable")
		
		i = 1 # [act, ind1, act, ind2, act, ind3]
		while i < len(message):
			if (int(self.name)-1) == int(message[i]):
				# discrete
				# action = int(message[i-1])
				# break

				# continuous
				temp_act_ind = i - 1
				# roll_inp = message[temp_act_ind]
				# temp_act_ind -= 1
				pitch_inp = message[temp_act_ind]
				temp_act_ind -= 1
				# yaw_inp = message[temp_act_ind]
				break
			i += 2
		# action = int(message[0])  # Convert the string message into an action integer
		time = self.robot.getTime()
		roll = self.imu.getRollPitchYaw()[0] + 3.14159 / 2.0
		pitch = self.imu.getRollPitchYaw()[1]
		altitude = self.gps.getValues()[1]
		# self.target_altitude = altitude
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
		
		if pitch_inp == 'None':
			pitch_disturbance = 0.0
			# rot_val = np.random.uniform(-1, 0, 1)
			# toss = np.random.uniform(-1, 1, 1)
			# if toss <= 0.5:
			# 	the_val = rot_val
			# else:
			# 	the_val = rot_val / 2
			# yaw_disturbance = float(the_val)
		else:
			# pitch_disturbance = 1.0
			pitch_disturbance = float(pitch_inp)
			# yaw_disturbance = float(yaw_inp)

		altitude = self.gps.getValues()[1]

		# roll_disturbance = float(roll_inp) * 2
		# pitch_disturbance = float(pitch_inp) * 2
		# yaw_disturbance = float(yaw_inp)

		# if abs(altitude - self.target_altitude) <= 0.5:
		# 	if action == 0:
		# 		pitch_disturbance = 1.0
		# 	elif action == 1:
		# 		pitch_disturbance = -1.0
		# 	elif action == 2:
		# 		yaw_disturbance = 1.0
		# 	elif action == 3:
		# 		yaw_disturbance = -1.0
			# elif action == 4:
			# 	roll_disturbance = -1.0
			# elif action == 5:
			# 	roll_disturbance = 1.0
			# elif action == 4:
			# 	self.target_altitude += 0.05
			# elif action == 5:
			# 	self.target_altitude -= 0.05
		
		# Set the motors' velocities based on the action received
		roll_input = self.k_roll_p * self.CLAMP(roll, -1.0, 1.0) + roll_acceleration + roll_disturbance
		pitch_input = self.k_pitch_p * self.CLAMP(pitch, -1.0, 1.0) - pitch_acceleration + pitch_disturbance
		yaw_input = yaw_disturbance
		clamped_difference_altitude = self.CLAMP(self.target_altitude - altitude + self.k_vertical_offset, -1.0, 1.0)
		vertical_input = self.k_vertical_p * pow(clamped_difference_altitude, 3.0)
		
		# print('roll_input', roll_input, 'pitch_input', pitch_input, 'yaw_input', yaw_input)
		
		front_left_motor_input = self.k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
		front_right_motor_input = self.k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
		rear_left_motor_input = self.k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
		rear_right_motor_input = self.k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input

		# print(self.name)
		# print('front_left_motor', front_left_motor_input)
		# print('front_right_motor', front_right_motor_input)
		# print('rear_left_motor', rear_left_motor_input)
		# print('rear_right_motor', rear_right_motor_input)

		self.front_left_motor.setVelocity(front_left_motor_input)
		self.front_right_motor.setVelocity(-front_right_motor_input)
		self.rear_left_motor.setVelocity(-rear_left_motor_input)
		self.rear_right_motor.setVelocity(rear_right_motor_input)

# Create the robot controller object and run it
robot_controller = CartpoleRobot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it
