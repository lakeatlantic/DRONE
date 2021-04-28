from controller import *
import mavic2proHelper
from simple_pid import PID
import csv
import struct
from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
import math
from abc import abstractmethod
from collections.abc import Iterable
import numpy as np

params = dict()
with open("../params.csv", "r") as f:
	lines = csv.reader(f)
	for line in lines:
		params[line[0]] = line[1]

TIME_STEP = int(params["QUADCOPTER_TIME_STEP"])
TAKEOFF_THRESHOLD_VELOCITY = int(params["TAKEOFF_THRESHOLD_VELOCITY"])
M_PI = 3.1415926535897932384626433

class CartpoleRobot(RobotEmitterReceiverCSV):
	def __init__(self):
		super().__init__()
		self.name = self.robot.getName()
		self.all_motors = mavic2proHelper.getMotorAll(self.robot)

		self.timestep = int(self.robot.getBasicTimeStep())
		self.mavic2proMotors = mavic2proHelper.getMotorAll(self.robot)
		mavic2proHelper.initialiseMotors(self.robot, 0)
		mavic2proHelper.motorsSpeed(self.robot, TAKEOFF_THRESHOLD_VELOCITY, TAKEOFF_THRESHOLD_VELOCITY, TAKEOFF_THRESHOLD_VELOCITY, TAKEOFF_THRESHOLD_VELOCITY)

		self.front_left_led = LED("front left led")
		self.front_right_led = LED("front right led")
		self.gps = GPS("gps")
		self.gps.enable(TIME_STEP)
		self.imu = InertialUnit("inertial unit")
		self.imu.enable(TIME_STEP)
		self.compass = Compass("compass")
		self.compass.enable(TIME_STEP)
		self.gyro = Gyro("gyro")
		self.gyro.enable(TIME_STEP)

		self.yaw_setpoint=-1

		self.pitchPID = PID(float(params["pitch_Kp"]), float(params["pitch_Ki"]), float(params["pitch_Kd"]), setpoint=0.0)
		self.rollPID = PID(float(params["roll_Kp"]), float(params["roll_Ki"]), float(params["roll_Kd"]), setpoint=0.0)
		self.throttlePID = PID(float(params["throttle_Kp"]), float(params["throttle_Ki"]), float(params["throttle_Kd"]), setpoint=1)
		self.yawPID = PID(float(params["yaw_Kp"]), float(params["yaw_Ki"]), float(params["yaw_Kd"]), setpoint=float(self.yaw_setpoint))
		
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
		
		i = 2 # [act, act, act, ind1, act, act, act, ind2]
		while i < len(message):
			if (int(self.name)-1) == int(message[i]):
				# continuous
				temp_act_ind = i - 1
				targetY = float(message[temp_act_ind])
				temp_act_ind -= 1
				# target_altitude = float(message[temp_act_ind])
				# temp_act_ind -= 1
				targetX = float(message[temp_act_ind])
				break
			i += 3
		
		# while self.robot.step(self.timestep) != -1:
		led_state = int(self.robot.getTime()) % 2
		self.front_left_led.set(led_state)
		self.front_right_led.set(int(not(led_state)))

		roll = self.imu.getRollPitchYaw()[0] + M_PI / 2.0
		pitch = self.imu.getRollPitchYaw()[1]
		yaw = self.compass.getValues()[0]
		roll_acceleration = self.gyro.getValues()[0]
		pitch_acceleration = self.gyro.getValues()[1]
		
		# print(self.gps.getValues())
		xGPS = self.gps.getValues()[2]
		yGPS = self.gps.getValues()[0]
		zGPS = self.gps.getValues()[1]

		self.throttlePID.setpoint = 1.0
		vertical_input = self.throttlePID(zGPS)
		yaw_input = self.yawPID(yaw)

		self.rollPID.setpoint = targetX
		self.pitchPID.setpoint = -1 * targetY
		
		roll_input = float(params["k_roll_p"]) * roll + roll_acceleration + self.rollPID(xGPS)
		pitch_input = float(params["k_pitch_p"]) * pitch - pitch_acceleration + self.pitchPID(-yGPS)

		front_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - roll_input - pitch_input + yaw_input
		front_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + roll_input - pitch_input - yaw_input
		rear_left_motor_input = float(params["k_vertical_thrust"]) + vertical_input - roll_input + pitch_input - yaw_input
		rear_right_motor_input = float(params["k_vertical_thrust"]) + vertical_input + roll_input + pitch_input + yaw_input

		mavic2proHelper.motorsSpeed(self.robot, front_left_motor_input, -front_right_motor_input, -rear_left_motor_input, rear_right_motor_input)


# Create the robot controller object and run it
robot_controller = CartpoleRobot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it
