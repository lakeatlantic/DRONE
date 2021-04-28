from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from abc import abstractmethod
from collections.abc import Iterable
import argparse
import torch
import time
import os
import copy
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from os import path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
import math
import csv
##
import os
import glob
import time
from datetime import datetime
import torch
import numpy as np
import gym
# import roboschool
# import pybullet_envs
from PPO import PPO

global_run_dir = None
num_agents = 1
global_et_i = 1
DIST_SENSORS_MM = {'min': -2.5, 'max': 2.5}

class CartPoleSupervisor(SupervisorCSV):
	def __init__(self):
		super().__init__()
		self.num_agents = num_agents

		self.robot = []
		self.target = []
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
			childrenField.importMFNode(-1, "mavic"+str(i+1)+".wbo")
			childrenField.importMFNode(-1, "Location"+str(i+1)+".wbo")

		# UAV respawn
		for i in range(num_agents):
			self.robot.append(self.supervisor.getFromDef("ROBOT"+str(i+1)))
			robot_node = self.supervisor.getFromDef("ROBOT"+str(i+1))
			# rn = self.supervisor.getFromDef("ROBOT"+str(i+1)+".BODY_SLOT")
			# cf = rn.getField('children')
			# cf.importMFNode(-1, "Solid"+str(i+1)+".wbo")
			trans_field = robot_node.getField("translation")
			pos = np.random.uniform(-5, 5, 3)
			pos[0] = 2
			pos[1] = 0.1
			pos[2] = 3

			location = pos.tolist()
			trans_field.setSFVec3f(location)
			robot_node.resetPhysics()

		# box respawn
		for i in range(num_agents):
			self.box.append(self.supervisor.getFromDef("LOC"+str(i+1)))
			box_node = self.supervisor.getFromDef("LOC"+str(i+1))
			trans_field = box_node.getField("translation")
			pos = np.random.uniform(-5, 5, 3)
			pos[0] = -2
			pos[1] = 0.001
			pos[2] = -3

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

	def normalize_to_range(self, value, min, max, newMin, newMax):
		value = float(value)
		min = float(min)
		max = float(max)
		newMin = float(newMin)
		newMax = float(newMax)
		return (newMax - newMin) / (max - min) * (value - max) + newMax


	def get_observation_agent(self, i):
		target_pos = []
		target_pos.append(np.array(self.box[i].getPosition()) - np.array(self.robot[i].getPosition()))
		other_pos = []
		# for j in range(num_agents):
		# 	if j == i:
		# 		continue
		# 	other_pos.append(np.array(self.robot[j].getPosition()))

		own_pos = np.array(self.robot[i].getPosition())
		own_vel = np.array(self.robot[i].getVelocity())
		# own_ori = np.array(self.robot[i].getOrientation())
		concatenated = np.concatenate([own_pos] + [own_vel] + target_pos + other_pos)
		return concatenated

	def unit_vector(self, vector):
		return vector / np.linalg.norm(vector)

	def get_angle_from_target(self, robot_node, target_node, is_true_angle=False, is_abs=False):
		# both UAVs parallel here
		v1 = np.array([robot_node.getOrientation()[2], robot_node.getOrientation()[5], robot_node.getOrientation()[8]])
		v2 = np.array([target_node.getOrientation()[1], target_node.getOrientation()[4], target_node.getOrientation()[7]])
		v1_u = self.unit_vector(v1)
		v2_u = self.unit_vector(v2)
		return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

		# robotAngle = robot_node.getField('rotation').getSFRotation()[3]
		# robot = robot_node.getField('translation').getSFVec3f()
		# target = target_node.getField('translation').getSFVec3f()
		# robot[0], robot[1], robot[2] = robot[0]-target[0], robot[1]-target[1], robot[2]-target[2]
		# target[0], target[1], target[2] = target[0]-target[0], target[1]-target[1], target[2]-target[2]
		# robotCoordinates = robot
		# targetCoordinate = target

		# x_r = (targetCoordinate[0] - robotCoordinates[0])
		# z_r = (targetCoordinate[2] - robotCoordinates[2])

		# z_r = -z_r

		# # robotWorldAngle = math.atan2(robotCoordinates[2], robotCoordinates[0])

		# if robotAngle < 0.0: robotAngle += 2 * np.pi

		# x_f = x_r * math.sin(robotAngle) - \
		#       z_r * math.cos(robotAngle)

		# z_f = x_r * math.cos(robotAngle) + \
		#       z_r * math.sin(robotAngle)

		# # print("x_f: {} , z_f: {}".format(x_f, z_f) )
		# if is_true_angle:
		#     x_f = -x_f
		# angleDif = math.atan2(z_f, x_f)

		# if is_abs:
		#     angleDif = abs(angleDif)

		# return angleDif

	def get_reward_agent(self, i, action, pre_pos):
		one = np.array(self.box[i].getPosition())
		one[1] = 1.0
		two = np.array(self.robot[i].getPosition())
		diff_dist = np.sqrt(np.sum(np.square(one - two)))
		# if diff_dist < 4.0 and diff_dist > 3.0:
		# 	return 10
		# elif diff_dist < 3.0 and diff_dist > 2.0:
		# 	return 20
		# elif diff_dist < 2.0 and diff_dist > 1.0:
		# 	return 30
		# elif diff_dist < 1.0 and diff_dist > 0.5:
		# 	return 40
		# else:

		if diff_dist < 1:
			# if global_run_dir is not None:
			# 	with open(global_run_dir / 'logdemo.csv', 'a', newline='') as file__:
			# 		writerlog = csv.writer(file__)
			# 		writerlog.writerow([action[0], action[1], 10])
			return 10
		else:
			# if global_run_dir is not None:
			# 	with open(global_run_dir / 'logdemo.csv', 'a', newline='') as file__:
			# 		writerlog = csv.writer(file__)
			# 		writerlog.writerow([action[0], action[1], diff_dist])
			return -0.1 * diff_dist

	def is_done_agent(self, i):
		return False

	def get_info_agent(self, i):
		return {}

	##################
	##################
	##################
	##################

	def solved(self):
		return False

	def reset(self):
		self.respawnRobot()
		self.supervisor.simulationResetPhysics()  # Reset the simulation physics to start over
		
		self.messageReceived = None

		# obs_n = []
		# for i in range(num_agents):
		# 	obs_n.append(self.get_observation_agent(i))
		# single_env_obs = []
		# single_env_obs.append(obs_n)

		return self.get_observation_agent(0)

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

	def stepup(self):
		for i in range(num_agents):
			one = np.array(self.box[i].getField("translation").getSFVec3f())
			one[1] = 1.0
			two = np.array(self.robot[i].getField("translation").getSFVec3f())
			diff = np.sqrt(np.sum(np.square(one - two)))

			# if diff < 1:
			# 	print('picked')
			# 	box_node = self.supervisor.getFromDef("LOC"+str(i+1))
			# 	trans_field = box_node.getField("translation")
			# 	pos = np.random.uniform(-5, 5, 3)
			# 	pos[1] = 0.001

			# 	location = pos.tolist()
			# 	trans_field.setSFVec3f(location)
			# 	box_node.resetPhysics()

	def step(self, action):
		if self.supervisor.step(self.timestep) == -1:
			exit()
		# print('action', action)
		actind_i = []
		for i in range(num_agents):
			# discrete
			# result = np.where(action[0][i] == np.amax(action[0][i]))
			# ind = result[0][0]
			# actind_i.append(ind)
			# actind_i.append(i)

			# continuous
			for j in action:
				actind_i.append(j)
			actind_i.append(i)

		pre_pos = []
		for i in range(num_agents):
			pre_pos.append(np.array(self.robot[i].getField("translation").getSFVec3f()))		
		
		self.handle_emitter(actind_i)
		# obs_n = []
		# for i in range(num_agents):
		# 	obs_n.append(self.get_observation_agent(i))
		# single_env_obs = []
		# single_env_obs.append(obs_n)
		# rews_n = []
		# for i in range(num_agents):
		# 	rews_n.append(self.get_reward_agent(i, action[0][i], pre_pos[i]))
		# dones_n = []
		# for i in range(num_agents):
		# 	dones_n.append(self.is_done_agent(i))
		# infos_n = {'n': []}
		# for i in range(num_agents):
		# 	infos_n['n'].append(self.get_info_agent(i))

		# self.stepup()

		# single_env_obs = []
		# single_env_obs.append(obs_n)
		# single_env_rews = []
		# single_env_rews.append(rews_n)
		# single_env_dones = []
		# single_env_dones.append(dones_n)
		# single_env_infos = []
		# single_env_infos.append(infos_n)
		return self.get_observation_agent(0), self.get_reward_agent(i, action, pre_pos[i]), self.is_done_agent(i), self.get_info_agent(i)

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

train = True
# train = False
that_one = True
# that_one = False
cont = True

if train:
	## initialize environment hyperparameter
	supervisor = CartPoleSupervisor()
	has_continuous_action_space = True
	max_ep_len = 1000
	max_training_timesteps = int(3e6)
	print_freq = max_ep_len * 10
	log_freq = max_ep_len * 2
	save_model_freq = int(1e5)
	action_std = 0.6
	action_std_decay_rate = 0.05
	min_action_std = 0.1 
	action_std_decay_freq = int(2.5e5)
	## PPO hyperparameters
	update_timestep = max_ep_len * 4
	K_epochs = 80
	eps_clip = 0.2
	gamma = 0.99
	lr_actor = 0.0003
	lr_critic = 0.001
	random_seed = 0
	env_name = 'UAV'
	print("training environment name : " + env_name)
	## state space dimension
	state_dim = 12
	## action space dimension
	if has_continuous_action_space:
		action_dim = 2
	else:
		action_dim = 5
	## log files for multiple runs are NOT overwritten
	log_dir = "PPO_logs"
	if not os.path.exists(log_dir):
		  os.makedirs(log_dir)
	log_dir = log_dir + '/' + env_name + '/'
	if not os.path.exists(log_dir):
		  os.makedirs(log_dir)
	## get number of log files in log directory
	run_num = 0
	current_num_files = next(os.walk(log_dir))[2]
	run_num = len(current_num_files)
	## create new log file for each run
	log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
	print("current logging run number for " + env_name + " : ", run_num)
	print("logging at : " + log_f_name)
	## checkpointing
	run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder
	directory = "PPO_preTrained"
	if not os.path.exists(directory):
		  os.makedirs(directory)
	directory = directory + '/' + env_name + '/'
	if not os.path.exists(directory):
		  os.makedirs(directory)
	checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
	print("save checkpoint path : " + checkpoint_path)
	## print all hyperparameters
	print("--------------------------------------------------------------------------------------------")
	print("max training timesteps : ", max_training_timesteps)
	print("max timesteps per episode : ", max_ep_len)
	print("model saving frequency : " + str(save_model_freq) + " timesteps")
	print("log frequency : " + str(log_freq) + " timesteps")
	print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
	print("--------------------------------------------------------------------------------------------")
	print("state space dimension : ", state_dim)
	print("action space dimension : ", action_dim)
	print("--------------------------------------------------------------------------------------------")
	if has_continuous_action_space:
		print("Initializing a continuous action space policy")
		print("--------------------------------------------------------------------------------------------")
		print("starting std of action distribution : ", action_std)
		print("decay rate of std of action distribution : ", action_std_decay_rate)
		print("minimum std of action distribution : ", min_action_std)
		print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
	else:
		print("Initializing a discrete action space policy")
	print("--------------------------------------------------------------------------------------------")
	print("PPO update frequency : " + str(update_timestep) + " timesteps")
	print("PPO K epochs : ", K_epochs)
	print("PPO epsilon clip : ", eps_clip)
	print("discount factor (gamma) : ", gamma)
	print("--------------------------------------------------------------------------------------------")
	print("optimizer learning rate actor : ", lr_actor)
	print("optimizer learning rate critic : ", lr_critic)
	if random_seed:
		print("--------------------------------------------------------------------------------------------")
		print("setting random seed to ", random_seed)
		torch.manual_seed(random_seed)
		env.seed(random_seed)
		np.random.seed(random_seed)
	print("============================================================================================")
	## training procedure
	## initialize a PPO agent
	ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
	## track total training time
	start_time = datetime.now().replace(microsecond=0)
	print("Started training at (GMT) : ", start_time)
	print("============================================================================================")
	## logging file
	log_f = open(log_f_name,"w+")
	log_f.write('episode,timestep,reward\n')
	## printing and logging variables
	print_running_reward = 0
	print_running_episodes = 0
	log_running_reward = 0
	log_running_episodes = 0
	time_step = 0
	i_episode = 0
	
	# training loop
	while time_step <= max_training_timesteps:
		state = supervisor.reset()
		current_ep_reward = 0
		for z in range(400):
			action = [0.0, 0.0]
			state, reward, done, _ = supervisor.step(action)
		for t in range(1, max_ep_len+1):
			if supervisor.check():
				break
			# select action with policy
			action = ppo_agent.select_action(state)
			state, reward, done, _ = supervisor.step(action)

			# saving reward and is_terminals
			ppo_agent.buffer.rewards.append(reward)
			ppo_agent.buffer.is_terminals.append(done)

			time_step +=1
			current_ep_reward += reward

			# update PPO agent
			if time_step % update_timestep == 0:
				ppo_agent.update()

			# if continuous action space; then decay action std of ouput action distribution
			if has_continuous_action_space and time_step % action_std_decay_freq == 0:
				ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

			# log in logging file
			if time_step % log_freq == 0:

				# log average reward till last episode
				log_avg_reward = log_running_reward / log_running_episodes
				log_avg_reward = round(log_avg_reward, 4)

				log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
				log_f.flush()

				log_running_reward = 0
				log_running_episodes = 0

			# printing average reward
			if time_step % print_freq == 0:

				# print average reward till last episode
				print_avg_reward = print_running_reward / print_running_episodes
				print_avg_reward = round(print_avg_reward, 2)

				print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

				print_running_reward = 0
				print_running_episodes = 0

			# save model weights
			if time_step % save_model_freq == 0:
				print("--------------------------------------------------------------------------------------------")
				print("saving model at : " + checkpoint_path)
				ppo_agent.save(checkpoint_path)
				print("model saved")
				print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
				print("--------------------------------------------------------------------------------------------")

			# break; if the episode is over
			if done:
				break

		print_running_reward += current_ep_reward
		print_running_episodes += 1
		log_running_reward += current_ep_reward
		log_running_episodes += 1
		i_episode += 1

	log_f.close()
	supervisor.close()

	# print total training time
	print("============================================================================================")
	end_time = datetime.now().replace(microsecond=0)
	print("Started training at (GMT) : ", start_time)
	print("Finished training at (GMT) : ", end_time)
	print("Total training time  : ", end_time - start_time)
	print("============================================================================================")

else:

	env_id = 'uav'
	model_name = 'maddpg'
	save_gifs = False
	run_num = 1
	incremental = None
	n_rollout_threads = 1

	n_episodes = 10
	episode_length = 2000

	# model_path = (Path('/home/arshdeep/models') / env_id / model_name / ('run%i' % run_num))
	model_path = (Path('/home/sagar/models') / env_id / model_name / ('run%i' % run_num))

	if incremental is not None:
		model_path = model_path / 'incremental' / ('model_ep%i.pt' % incremental)
	else:
		model_path = model_path / 'model.pt'

	supervisor = CartPoleSupervisor()
	# supervisor.load_locations()

	maddpg = MADDPG.init_from_save(model_path)
	maddpg.prep_rollouts(device='cpu')

	for ep_i in range(n_episodes):
		print("Episode %i of %i" % (ep_i + 1, n_episodes))
		obs = supervisor.reset()

		check_point = 400
		itm_et_i = 0
		
		# init actions
		temp_act_init = []
		for i in range(num_agents):
			if that_one:
				temp_act_init.append(np.array([0.0, 0.0]))
			else:
				temp_act_init.append(np.array([supervisor.robot[i].getField("translation").getSFVec3f()[2], supervisor.robot[i].getField("translation").getSFVec3f()[0]]))
		temp_actions = []
		temp_actions.append(temp_act_init)

		# rise to given altitude
		for t_i in range(check_point):
			supervisor.step_init(temp_actions)
			itm_et_i = t_i
		t_i = 0 # itm_et_i
		for t_i in range(episode_length):
			if supervisor.check():
				break
			# supervisor.flight(t_i)
			# rearrange observations to be per agent, and convert to torch Variable
			torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(maddpg.nagents)]
			# torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1), requires_grad=False) for i in range(maddpg.nagents)]

			torch_agent_actions = maddpg.step(torch_obs, explore=True)
			agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
			actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]

			if that_one:
				pass
			else:
				for i in range(num_agents):
					actions[0][i][0] = 2 * actions[0][i][0] + supervisor.robot[i].getField("translation").getSFVec3f()[2]
					actions[0][i][1] = 2 * actions[0][i][1] + supervisor.robot[i].getField("translation").getSFVec3f()[0]
			# for m in range(10):
			next_obs, rewards, dones, infos = supervisor.step(actions)

	supervisor.close()
