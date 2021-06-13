from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from abc import abstractmethod
from collections.abc import Iterable
import argparse
import time
import math
import csv
import copy
import numpy as np

import os
from os import path
from pathlib import Path
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG

num_agents = 1

class CartPoleSupervisor(SupervisorCSV):
	def __init__(self):
		super().__init__()
		self.num_agents = num_agents

		self.agent_u_noise = None
		self.agent_mass = 1.0
		self.agent_accel = 3.0 # 3.0
		self.agent_p_vel = np.array([0.0, 0.0])
		self.damping = 0.25
		self.dt = 0.8
		self.max_speed = 1.0

		self.robot = []
		self.box = []
		self.messageReceived = None	 # Variable to save the messages received from the robot

	# currently suspended
	def load_locations(self):
		rootNode = self.supervisor.getRoot()
		childrenField = rootNode.getField('children')
		for i in range(num_agents):
			childrenField.importMFNode(-1, "Location"+str(i+1)+".wbo")

		for i in range(num_agents):
			self.box.append(self.supervisor.getFromDef("LOC"+str(i+1)))

	def respawnRobot(self):
		if len(self.robot) != 0:
			# Despawn existing robot
			for i in range(num_agents):
				self.robot[i].remove()
				self.box[i].remove()
			self.robot.clear()
			self.box.clear()

		# Respawn robot in starting position and state
		rootNode = self.supervisor.getRoot()
		childrenField = rootNode.getField('children')
		for i in range(num_agents):
			childrenField.importMFNode(-1, "Robot"+str(i+1)+".wbo")
			childrenField.importMFNode(-1, "Location"+str(i+1)+".wbo")

		# UAV respawn
		for i in range(num_agents):
			self.robot.append(self.supervisor.getFromDef("ROBOT"+str(i+1)))
			robot_node = self.supervisor.getFromDef("ROBOT"+str(i+1))
			trans_field = robot_node.getField("translation")
			pos = np.random.uniform(-1, 1, 3)
			pos[0] = 0
			pos[1] = 0.1
			pos[2] = 0

			location = pos.tolist()
			trans_field.setSFVec3f(location)
			robot_node.resetPhysics()

		# box respawn
		for i in range(num_agents):
			self.box.append(self.supervisor.getFromDef("LOC"+str(i+1)))
			box_node = self.supervisor.getFromDef("LOC"+str(i+1))
			trans_field = box_node.getField("translation")
			pos = np.random.uniform(-1, 1, 3)
			pos[0] = 10
			pos[1] = 0.0001
			pos[2] = 8

			location = pos.tolist()
			trans_field.setSFVec3f(location)
			box_node.resetPhysics()

	def check(self):
		for i in range(num_agents):
			custom_val = self.supervisor.getFromDef("ROBOT"+str(i+1)).getField("customData").getSFString()
			if custom_val == "unstable":
				return True
		return False

	def close(self):
		if len(self.robot) != 0:
			# Despawn existing robot
			for i in range(num_agents):
				self.robot[i].remove()
			self.robot.clear()
		if len(self.box) != 0:
			# Despawn existing robot
			for i in range(num_agents):
				self.box[i].remove()
			self.box.clear()

	#####################
	#####################

	def flight(self, t_i):
		print('timestep ==>', t_i)
		for i in range(num_agents):
			# print('height of UAV', i, 'is', self.robot[i].getField("translation").getSFVec3f()[1])
			print('height of UAV', i, 'is', self.robot[i].getPosition()[1])

	def get_observations(self):
		pass

	def get_reward(self, action=None):
		pass

	def is_done(self):
		pass

	def get_info(self):
		pass

	##################
	##################
	##################
	##################

	def get_observation_agent(self, i):
		target_pos = []
		target_pos.append(np.array(self.box[i].getPosition()) - np.array(self.robot[i].getPosition()))
		target_pos_f = np.array([target_pos[0][0], target_pos[0][2]])

		own_pos = np.array(self.robot[i].getPosition())
		own_pos_f = np.array([own_pos[0], own_pos[2]])

		concatenated = np.concatenate([own_pos_f] + [target_pos_f])
		return concatenated

	def get_reward_agent(self, i, action, pre_pos):
		rew = 0
		uav = np.array(self.robot[i].getPosition())
		uav[1] = 1.0
		tar = np.array(self.box[i].getPosition())
		tar[1] = 1.0
		rew -= 0.1 * np.sqrt(np.sum(np.square(uav - tar)))
		if self.is_hover(uav, tar):
			rew += 10
		return rew

	def is_done_agent(self, i):
		return False

	def get_info_agent(self, i):
		return {}

	##################
	##################
	##################
	##################

	def is_hover(self, agent1pos, agent2pos):
		delta_pos = agent1pos - agent2pos
		dist = np.sqrt(np.sum(np.square(delta_pos)))
		dist_min = 0.03 + 0.02 # uav radius + target radius
		return True if dist < dist_min else False

	def solved(self):
		return False

	def reset(self):
		self.respawnRobot()
		self.supervisor.simulationResetPhysics()  # Reset the simulation physics to start over
		
		self.messageReceived = None

		obs_n = []
		for i in range(num_agents):
			obs_n.append(self.get_observation_agent(i))
		single_env_obs = []
		single_env_obs.append(obs_n)

		return np.array(single_env_obs)

	def step_init(self, action):
		if self.supervisor.step(self.timestep) == -1:
			exit()

		actind_i = []
		for i in range(num_agents):
			# discrete
			# result = np.where(action[0][i] == np.amax(action[0][i]))
			# ind = result[0][0]
			# actind_i.append(-1)
			# actind_i.append(i)

			# continuous
			for j in action[0][i]:
				actind_i.append(j)
			actind_i.append(i)

		self.handle_emitter(actind_i)

	def poststep(self):
		for i in range(num_agents):
			uav = np.array(self.robot[i].getField("translation").getSFVec3f())
			uav[1] = 1.0
			tar = np.array(self.box[i].getField("translation").getSFVec3f())
			tar[1] = 1.0
			if self.is_hover(uav, tar):
				pass
				# self.box.append(self.supervisor.getFromDef("LOC"+str(i+1)))
				# box_node = self.supervisor.getFromDef("LOC"+str(i+1))
				# trans_field = box_node.getField("translation")
				# pos = np.random.uniform(-1, 1, 3)
				# # pos[0] = -2
				# pos[1] = 0.0001
				# # pos[2] = -2

				# location = pos.tolist()
				# trans_field.setSFVec3f(location)
				# box_node.resetPhysics()

	def step(self, action):
		if self.supervisor.step(self.timestep) == -1:
			exit()

		# dynamics start
		p_force = [None] * (self.num_agents)
		for i,agent in enumerate(self.robot):
			noise = np.random.randn(2) * self.agent_u_noise if self.agent_u_noise else 0.0
			p_force[i] = (self.agent_mass * self.agent_accel if self.agent_accel is not None else 
				self.agent_mass) * action[0][i] + noise

		for i,agent in enumerate(self.robot):
			self.agent_p_vel = self.agent_p_vel * (1 - self.damping)
			if (p_force[i] is not None):
				self.agent_p_vel += (p_force[i] / self.agent_mass) * self.dt
			if self.max_speed is not None:
				speed = np.sqrt(np.square(self.agent_p_vel[0]) + np.square(self.agent_p_vel[1]))
				if speed > self.max_speed:
					self.agent_p_vel = self.agent_p_vel / np.sqrt(np.square(self.agent_p_vel[0]) +
																  np.square(self.agent_p_vel[1])) * self.max_speed
			action[0][i][0] = self.robot[i].getField("translation").getSFVec3f()[2] + self.agent_p_vel[0] * self.dt
			action[0][i][1] = self.robot[i].getField("translation").getSFVec3f()[0] + self.agent_p_vel[1] * self.dt
		# dynamics end

		actind_i = []
		for i in range(num_agents):
			# discrete
			# result = np.where(action[0][i] == np.amax(action[0][i]))
			# ind = result[0][0]
			# actind_i.append(ind)
			# actind_i.append(i)

			# continuous
			for j in action[0][i]:
				actind_i.append(j)
			actind_i.append(i)

		pre_pos = []
		for i in range(num_agents):
			pre_pos.append(np.array(self.robot[i].getField("translation").getSFVec3f()))		
		
		self.handle_emitter(actind_i)
		obs_n = []
		for i in range(num_agents):
			obs_n.append(self.get_observation_agent(i))
		single_env_obs = []
		single_env_obs.append(obs_n)
		rews_n = []
		for i in range(num_agents):
			rews_n.append(self.get_reward_agent(i, action[0][i], pre_pos[i]))
		dones_n = []
		for i in range(num_agents):
			dones_n.append(self.is_done_agent(i))
		infos_n = {'n': []}
		for i in range(num_agents):
			infos_n['n'].append(self.get_info_agent(i))

		self.poststep()

		single_env_obs = []
		single_env_obs.append(obs_n)
		single_env_rews = []
		single_env_rews.append(rews_n)
		single_env_dones = []
		single_env_dones.append(dones_n)
		single_env_infos = []
		single_env_infos.append(infos_n)
		return np.array(single_env_obs), np.array(single_env_rews), np.array(single_env_dones), single_env_infos

	def handle_emitter(self, action):
		assert isinstance(action, Iterable), \
			"The action object should be Iterable"
		# action.append(i)
		message = (",".join(map(str, action))).encode("utf-8")
		self.emitter.send(message)

#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################

env = CartPoleSupervisor()
obs = env.reset()

# init actions
temp_act_init = []
for i in range(num_agents):
	temp_act_init.append(np.array([0.0, 0.0]))
temp_actions = []
temp_actions.append(temp_act_init)

# rise to given altitude
for et_i in range(400):
	env.step_init(temp_actions)
	
import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


def dwa_control(x, config, goal, ob):
	"""
	Dynamic Window Approach control
	"""
	dw = calc_dynamic_window(x, config)

	u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

	return u, trajectory


class RobotType(Enum):
	circle = 0
	rectangle = 1


class Config:
	"""
	simulation parameter class
	"""

	def __init__(self):
		# robot parameter
		self.max_speed = 1.0  # [m/s]
		self.min_speed = -0.5  # [m/s]
		self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
		self.max_accel = 0.2  # [m/ss]
		self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
		self.v_resolution = 0.01  # [m/s]
		self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
		self.dt = 0.1  # [s] Time tick for motion prediction
		self.predict_time = 3.0  # [s]
		self.to_goal_cost_gain = 0.15
		self.speed_cost_gain = 1.0
		self.obstacle_cost_gain = 1.0
		self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
		self.robot_type = RobotType.circle

		# if robot_type == RobotType.circle
		# Also used to check if goal is reached in both types
		self.robot_radius = 1.0  # [m] for collision check

		# if robot_type == RobotType.rectangle
		self.robot_width = 0.5  # [m] for collision check
		self.robot_length = 1.2  # [m] for collision check
		# obstacles [x(m) y(m), ....]
		self.ob = np.array([[-1, -1],
							[0, 2],
							[4.0, 2.0],
							[5.0, 4.0],
							[5.0, 5.0],
							[5.0, 6.0],
							[5.0, 9.0],
							[8.0, 9.0],
							[7.0, 9.0],
							[8.0, 10.0],
							[9.0, 11.0],
							[12.0, 13.0],
							[12.0, 12.0],
							[15.0, 15.0],
							[13.0, 13.0]
							])

	@property
	def robot_type(self):
		return self._robot_type

	@robot_type.setter
	def robot_type(self, value):
		if not isinstance(value, RobotType):
			raise TypeError("robot_type must be an instance of RobotType")
		self._robot_type = value


config = Config()


def motion(x, u, dt):
	"""
	motion model
	"""

	x[2] += u[1] * dt
	x[0] += u[0] * math.cos(x[2]) * dt
	x[1] += u[0] * math.sin(x[2]) * dt
	x[3] = u[0]
	x[4] = u[1]

	return x


def calc_dynamic_window(x, config):
	"""
	calculation dynamic window based on current state x
	"""

	# Dynamic window from robot specification
	Vs = [config.min_speed, config.max_speed,
		  -config.max_yaw_rate, config.max_yaw_rate]

	# Dynamic window from motion model
	Vd = [x[3] - config.max_accel * config.dt,
		  x[3] + config.max_accel * config.dt,
		  x[4] - config.max_delta_yaw_rate * config.dt,
		  x[4] + config.max_delta_yaw_rate * config.dt]

	#  [v_min, v_max, yaw_rate_min, yaw_rate_max]
	dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
		  max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

	return dw


def predict_trajectory(x_init, v, y, config):
	"""
	predict trajectory with an input
	"""

	x = np.array(x_init)
	trajectory = np.array(x)
	time = 0
	while time <= config.predict_time:
		x = motion(x, [v, y], config.dt)
		trajectory = np.vstack((trajectory, x))
		time += config.dt

	return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
	"""
	calculation final input with dynamic window
	"""

	x_init = x[:]
	min_cost = float("inf")
	best_u = [0.0, 0.0]
	best_trajectory = np.array([x])

	# evaluate all trajectory with sampled input in dynamic window
	for v in np.arange(dw[0], dw[1], config.v_resolution):
		for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

			trajectory = predict_trajectory(x_init, v, y, config)
			# calc cost
			to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
			speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
			ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

			final_cost = to_goal_cost + speed_cost + ob_cost

			# search minimum trajectory
			if min_cost >= final_cost:
				min_cost = final_cost
				best_u = [v, y]
				best_trajectory = trajectory
				if abs(best_u[0]) < config.robot_stuck_flag_cons \
						and abs(x[3]) < config.robot_stuck_flag_cons:
					# to ensure the robot do not get stuck in
					# best v=0 m/s (in front of an obstacle) and
					# best omega=0 rad/s (heading to the goal with
					# angle difference of 0)
					best_u[1] = -config.max_delta_yaw_rate
	return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
	"""
	calc obstacle cost inf: collision
	"""
	ox = ob[:, 0]
	oy = ob[:, 1]
	dx = trajectory[:, 0] - ox[:, None]
	dy = trajectory[:, 1] - oy[:, None]
	r = np.hypot(dx, dy)

	if config.robot_type == RobotType.rectangle:
		yaw = trajectory[:, 2]
		rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
		rot = np.transpose(rot, [2, 0, 1])
		local_ob = ob[:, None] - trajectory[:, 0:2]
		local_ob = local_ob.reshape(-1, local_ob.shape[-1])
		local_ob = np.array([local_ob @ x for x in rot])
		local_ob = local_ob.reshape(-1, local_ob.shape[-1])
		upper_check = local_ob[:, 0] <= config.robot_length / 2
		right_check = local_ob[:, 1] <= config.robot_width / 2
		bottom_check = local_ob[:, 0] >= -config.robot_length / 2
		left_check = local_ob[:, 1] >= -config.robot_width / 2
		if (np.logical_and(np.logical_and(upper_check, right_check),
						   np.logical_and(bottom_check, left_check))).any():
			return float("Inf")
	elif config.robot_type == RobotType.circle:
		if np.array(r <= config.robot_radius).any():
			return float("Inf")

	min_r = np.min(r)
	return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
	"""
		calc to goal cost with angle difference
	"""

	dx = goal[0] - trajectory[-1, 0]
	dy = goal[1] - trajectory[-1, 1]
	error_angle = math.atan2(dy, dx)
	cost_angle = error_angle - trajectory[-1, 2]
	cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

	return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
	plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
			  head_length=width, head_width=width)
	plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
	if config.robot_type == RobotType.rectangle:
		outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
							 (config.robot_length / 2), -config.robot_length / 2,
							 -config.robot_length / 2],
							[config.robot_width / 2, config.robot_width / 2,
							 - config.robot_width / 2, -config.robot_width / 2,
							 config.robot_width / 2]])
		Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
						 [-math.sin(yaw), math.cos(yaw)]])
		outline = (outline.T.dot(Rot1)).T
		outline[0, :] += x
		outline[1, :] += y
		plt.plot(np.array(outline[0, :]).flatten(),
				 np.array(outline[1, :]).flatten(), "-k")
	elif config.robot_type == RobotType.circle:
		circle = plt.Circle((x, y), config.robot_radius, color="b")
		plt.gcf().gca().add_artist(circle)
		out_x, out_y = (np.array([x, y]) +
						np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
		plt.plot([x, out_x], [y, out_y], "-k")


def main(gx=10.0, gy=8.0, robot_type=RobotType.circle):
	print(__file__ + " start!!")
	# initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
	x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
	# goal position [x(m), y(m)]
	goal = np.array([gx, gy])

	# input [forward speed, yaw_rate]

	config.robot_type = robot_type
	trajectory = np.array(x)
	ob = config.ob
	while True:
		u, predicted_trajectory = dwa_control(x, config, goal, ob)
		print('u', u)
		# print('predicted_trajectory', len(predicted_trajectory))

		# init actions
		temp_act_init = []
		for i in range(num_agents):
			temp_act_init.append(np.array([u[0], u[1]]))
		temp_actions = []
		temp_actions.append(temp_act_init)

		# rise to given altitude
		for et_i in range(100):
			env.step_init(temp_actions)

		x = motion(x, u, config.dt)  # simulate robot
		trajectory = np.vstack((trajectory, x))  # store state history

		# if show_animation:
		# 	plt.cla()
		# 	# for stopping simulation with the esc key.
		# 	plt.gcf().canvas.mpl_connect(
		# 		'key_release_event',
		# 		lambda event: [exit(0) if event.key == 'escape' else None])
		# 	plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
		# 	plt.plot(x[0], x[1], "xr")
		# 	plt.plot(goal[0], goal[1], "xb")
		# 	plt.plot(ob[:, 0], ob[:, 1], "ok")
		# 	plot_robot(x[0], x[1], x[2], config)
		# 	plot_arrow(x[0], x[1], x[2])
		# 	plt.axis("equal")
		# 	plt.grid(True)
		# 	plt.pause(0.0001)

		# check reaching goal
		dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
		if dist_to_goal <= config.robot_radius:
			print("Goal!!")
			break

	print("Done")
	if show_animation:
		plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
		plt.pause(0.0001)

	plt.show()


# if __name__ == '__main__':
# main(robot_type=RobotType.rectangle)
main(robot_type=RobotType.circle)