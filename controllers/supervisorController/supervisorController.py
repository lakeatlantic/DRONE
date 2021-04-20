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

num_agents = 2
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
			pos[1] = 0.1

			location = pos.tolist()
			trans_field.setSFVec3f(location)
			robot_node.resetPhysics()

		# box respawn
		for i in range(num_agents):
			self.box.append(self.supervisor.getFromDef("LOC"+str(i+1)))
			box_node = self.supervisor.getFromDef("LOC"+str(i+1))
			trans_field = box_node.getField("translation")
			pos = np.random.uniform(-5, 5, 3)
			pos[1] = 0.001

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
		target_pos.append(np.array(self.box[i].getPosition()))
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
		two = np.array(self.robot[i].getPosition())
		diff_dist = np.sqrt(np.sum(np.square(one - two)))
		return -diff_dist

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

		actind_i = []
		for i in range(num_agents):
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

		self.stepup()

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

train = True
# train = False
that_one = True
# that_one = False
cont = True

if train:

	env_id = 'uav'
	model_name = 'maddpg'
	seed = 1
	n_training_threads = 1
	agent_alg = 'MADDPG'
	adversary_alg = 'MADDPG'
	tau = 0.01 #0.01
	lr = 0.01 #0.01
	hidden_dim = 64 #64
	buffer_length = int(1e6) #int(1e6)
	n_rollout_threads = 1
	n_exploration_eps = 25000 
	final_noise_scale = 0.0
	init_noise_scale = 0.3 #0.3
	batch_size = 1024 #1024, 32640
	steps_per_update = 100
	save_interval = 500

	n_episodes = 10000
	if that_one:
		episode_length = 1920 #1920
	else:
		episode_length = 2500 #1920

	USE_CUDA = False
	is_new_file = False

	model_dir = Path('/home/sagar/models') / env_id / model_name
	# model_dir = Path('/home/arshdeep/models') / env_id / model_name

	if not model_dir.exists():
		curr_run = 'run1'
		run_dir = model_dir / curr_run
		os.makedirs(run_dir)
		log_dir = run_dir / 'logs'
		os.makedirs(log_dir)
		file = open(run_dir / 'demo.txt', "w+")
		file.seek(0)
		file.write('0')

		file_t = open(run_dir / 'tboard.txt', "w+")
		file_t.seek(0)
		file_t.write('0')
		is_new_file = True
	else:
		exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
						 model_dir.iterdir() if
						 str(folder.name).startswith('run')]
		if len(exst_run_nums) == 0:
			curr_run = 'run1'
			run_dir = model_dir / curr_run
			os.makedirs(run_dir)
			log_dir = run_dir / 'logs'
			os.makedirs(log_dir)
			file = open(run_dir / 'demo.txt', "w+")
			file.seek(0)
			file.write('0')

			file_t = open(run_dir / 'tboard.txt', "w+")
			file_t.seek(0)
			file_t.write('0')
			is_new_file = True
		else:
			##
			curr_run = 'run%i' % (max(exst_run_nums) + 0)
			run_dir = model_dir / curr_run
			if path.exists(run_dir / 'demo.txt') == True:
				file = open(run_dir / 'demo.txt', "r+")
				file.seek(0)
				content = file.read()
				if int(content) + 1 < n_episodes:
					curr_run = 'run%i' % (max(exst_run_nums) + 0)
					run_dir = model_dir / curr_run
					log_dir = run_dir / 'logs'
					file = open(run_dir / 'demo.txt', "r+")
					file.seek(0)

					file_t = open(run_dir / 'tboard.txt', "r+")
					file_t.seek(0)
				else:
					curr_run = 'run%i' % (max(exst_run_nums) + 1)
					run_dir = model_dir / curr_run
					os.makedirs(run_dir)
					log_dir = run_dir / 'logs'
					os.makedirs(log_dir)
					file = open(run_dir / 'demo.txt', "w+")
					file.seek(0)
					file.write('0')

					file_t = open(run_dir / 'tboard.txt', "w+")
					file_t.seek(0)
					file_t.write('0')
					is_new_file = True
			else:
				curr_run = 'run%i' % (max(exst_run_nums) + 1)
				run_dir = model_dir / curr_run
				file = open(run_dir / 'demo.txt', "w+")
				file.seek(0)
				file.write('0')

				file_t = open(run_dir / 'tboard.txt', "w+")
				file_t.seek(0)
				file_t.write('0')
				is_new_file = True
			##
	logger = SummaryWriter(str(log_dir))

	torch.manual_seed(seed)
	np.random.seed(seed)
	if not USE_CUDA:
		torch.set_num_threads(n_training_threads)

	supervisor = CartPoleSupervisor()
	# supervisor.load_locations()

	if is_new_file == True:
		maddpg = MADDPG.init_from_env(supervisor, agent_alg=agent_alg, adversary_alg=adversary_alg, tau=tau, lr=lr, hidden_dim=hidden_dim)
	else:
		model_path = run_dir / 'model.pt'
		maddpg = MADDPG.init_from_save(model_path)

	obsp, acsp = [], []
	for i in range(num_agents):
		obsp.append(12)
		acsp.append(2)

	replay_buffer = ReplayBuffer(buffer_length, supervisor.num_agents, obsp, acsp)
	t = 0
	file.seek(0)
	index_val = int(file.read())

	file_t.seek(0)
	index_val_t = int(file_t.read())
	
	for ep_i in range(index_val, n_episodes, n_rollout_threads):
		file.seek(0)
		file.write(str(ep_i))
		print("Episodes %i-%i of %i" % (ep_i + 1, ep_i + 1 + n_rollout_threads, n_episodes))
		obs = supervisor.reset()
		maddpg.prep_rollouts(device='cpu')

		explr_pct_remaining = max(0, n_exploration_eps - ep_i) / n_exploration_eps
		maddpg.scale_noise(final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)
		maddpg.reset_noise()

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
		for et_i in range(check_point):
			# supervisor.flight(et_i)
			supervisor.step_init(temp_actions)
			itm_et_i = et_i
		et_i = 0 # itm_et_i
		for et_i in range(episode_length):
			if supervisor.check():
				break
			torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(maddpg.nagents)]
			torch_agent_actions = maddpg.step(torch_obs, explore=True)
			agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
			actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]
			global_et_i = et_i+1

			if that_one:
				pass
			else:
				for i in range(num_agents):
					actions[0][i][0] = 2 * actions[0][i][0] + supervisor.robot[i].getField("translation").getSFVec3f()[2]
					actions[0][i][1] = 2 * actions[0][i][1] + supervisor.robot[i].getField("translation").getSFVec3f()[0]
			next_obs, rewards, dones, infos = supervisor.step(actions)

			index_val_t += 1
			file_t.seek(0)
			file_t.write(str(index_val_t))
			# for a_i, pitch in enumerate(actions[0]):
			# 	logger.add_scalar('agent%i/Pitch' % a_i, pitch, index_val_t)
			# for a_i, yaw in enumerate(actions[0]):
			# 	logger.add_scalar('agent%i/Yaw' % a_i, yaw[1], index_val_t)

			replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
			obs = next_obs
			t += n_rollout_threads
			if (len(replay_buffer) >= batch_size and (t % steps_per_update) < n_rollout_threads):
				if USE_CUDA:
					maddpg.prep_training(device='gpu')
				else:
					maddpg.prep_training(device='cpu')
				for u_i in range(n_rollout_threads):
					for a_i in range(maddpg.nagents):
						sample = replay_buffer.sample(batch_size, to_gpu=USE_CUDA)
						maddpg.update(sample, index_val_t, a_i, logger=logger)
					maddpg.update_all_targets()
				maddpg.prep_rollouts(device='cpu')

		ep_rews = replay_buffer.get_average_rewards(episode_length * n_rollout_threads)
		for a_i, a_ep_rew in enumerate(ep_rews):
			logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

		if ep_i % save_interval < n_rollout_threads:
			os.makedirs(run_dir / 'incremental', exist_ok=True)
			maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
			maddpg.save(run_dir / 'model.pt')

	maddpg.save(run_dir / 'model.pt')
	logger.export_scalars_to_json(str(log_dir / 'summary.json'))
	logger.close()
	file.close()
	file_t.close()

	supervisor.close()

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
	supervisor.load_locations()

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
			next_obs, rewards, dones, infos = supervisor.step(actions)

	supervisor.close()
