import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from PPOAgent import PPOAgent, Transition
from utilities import normalizeToRange
import time
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

num_agents = 2
global_et_i = 1

class CartPoleSupervisor(SupervisorCSV):
	def __init__(self):
		super().__init__()
		# self.observationSpace = 3
		# self.actionSpace = 2
		self.num_agents = num_agents

		self.robot = []
		self.target = []
		# for i in range(num_agents):
		# 	pos = np.random.uniform(-20, 20, 3)
		# 	pos[1] = 0
		# 	self.target.append(pos)
		self.box = []
		# self.respawnRobot()
		# self.poleEndpoint = self.supervisor.getFromDef("POLE_ENDPOINT")
		self.messageReceived = None	 # Variable to save the messages received from the robot

		# self.episodeCount = 0  # Episode counter
		# self.episodeLimit = 100  # Max number of episodes allowed
		# self.stepsPerEpisode = 2000  # Max number of steps per episode
		# self.episodeScore = 0  # Score accumulated during an episode
		# self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved

	def load_locations(self):
		rootNode = self.supervisor.getRoot()  # This gets the root of the scene tree
		childrenField = rootNode.getField('children')  # This gets a list of all the children, ie. objects of the scene
		for i in range(num_agents):
			childrenField.importMFNode(-1, "Location"+str(i+1)+".wbo")

		for i in range(num_agents):
			self.box.append(self.supervisor.getFromDef("LOC"+str(i+1)))

	def respawnRobot(self):
		if len(self.robot) != 0:
			# Despawn existing robot
			for i in range(num_agents):
				self.robot[i].remove()
			self.robot.clear()
		# if len(self.box) != 0:
		# 	# Despawn existing robot
		# 	for i in range(num_agents):
		# 		self.box[i].remove()
		# 	self.box.clear()

		# Respawn robot in starting position and state
		rootNode = self.supervisor.getRoot()
		childrenField = rootNode.getField('children')
		for i in range(num_agents):
			childrenField.importMFNode(-1, "Robot"+str(i+1)+".wbo")
		# for i in range(num_agents):
		# 	childrenField.importMFNode(-1, "Location"+str(i+1)+".wbo")

		# Get the new robot and pole endpoint references
		for i in range(num_agents):
			self.robot.append(self.supervisor.getFromDef("ROBOT"+str(i+1)))
			robot_node = self.supervisor.getFromDef("ROBOT"+str(i+1))

			# rn = self.supervisor.getFromDef("ROBOT"+str(i+1)+".BODY_SLOT")
			# cf = rn.getField('children')
			# cf.importMFNode(-1, "Solid"+str(i+1)+".wbo")

			trans_field = robot_node.getField("translation")
			pos = np.random.uniform(0, 5, 3)
			pos[1] = 0.1
			location = pos.tolist()
			trans_field.setSFVec3f(location)
			robot_node.resetPhysics()

		for i in range(num_agents):
			# self.box.append(self.supervisor.getFromDef("LOC"+str(i+1)))
			box_node = self.supervisor.getFromDef("LOC"+str(i+1))
			trans_field = box_node.getField("translation")
			pos = np.random.uniform(0, 5, 3)
			pos[1] = 0.001

			location = pos.tolist()
			trans_field.setSFVec3f(location)
			box_node.resetPhysics()

		# self.target.clear()
		# for i in range(num_agents):
		# 	# self.box.append(self.supervisor.getFromDef("LOC"+str(i+1)))
		# 	box_node = self.supervisor.getFromDef("LOC"+str(i+1))
		# 	trans_field = box_node.getField("translation")
		# 	pos = np.random.uniform(0, 5, 3)
		# 	pos[1] = 0.001
		# 	temp_pos = copy.copy(pos)
		# 	temp_pos[1] = 1.0

		# 	location = pos.tolist()
		# 	trans_field.setSFVec3f(location)
		# 	box_node.resetPhysics()
		# 	self.target.append(temp_pos)

		# self.poleEndpoint = self.supervisor.getFromDef("POLE_ENDPOINT")

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

	def flight(self, t_i):
		print('timestep ==>', t_i)
		for i in range(num_agents):
			# print('height of UAV', i, 'is', self.robot[i].getField("translation").getSFVec3f()[1])
			print('height of UAV', i, 'is', self.robot[i].getPosition()[1])

	def get_observations(self):
		pass
		# obs = self.robot.getField("translation").getSFVec3f()
		# obs = self.robot.getPosition()
		# return obs

	def get_reward(self, action=None):
		pass
		# height = self.robot.getField("translation").getSFVec3f()[1]
		# height = self.robot.getPosition()[1]
		# diff = abs(1-height)
		# return -diff

	def is_done(self):
		pass
		# return False

	def get_info(self):
		pass
		# return None

	##################
	##################
	##################
	##################

	# def get_observation_agent(self, i):
	# 	# target_pos = []
	# 	# # target_pos.append(np.array(self.target[i]) - np.array(self.robot[i].getField("translation").getSFVec3f()))
	# 	# target_pos.append(np.array(self.target[i]))

	# 	target_pos = []
	# 	# target_pos.append(np.array(self.target[i]) - np.array(self.robot[i].getField("translation").getSFVec3f()))
	# 	temp_tar = []
	# 	temp_tar.append(self.target[i][0])
	# 	temp_tar.append(self.target[i][2])
	# 	target_pos.append(np.array(temp_tar))

	# 	other_pos = []
	# 	for j in range(num_agents):
	# 		if j == i:
	# 			continue
	# 		# # other_pos.append(np.array(self.robot[j].getField("translation").getSFVec3f()) - np.array(self.robot[i].getField("translation").getSFVec3f()))
	# 		# other_pos.append(np.array(self.robot[j].getPosition()))

	# 		other_temp = []
	# 		other_temp.append(np.array(self.robot[j].getPosition())[0])
	# 		other_temp.append(np.array(self.robot[j].getPosition())[2])
	# 		target_pos.append(np.array(other_temp))

	# 	# # concatenated = np.concatenate([np.array(self.robot[i].getField("translation").getSFVec3f())] + [np.array(self.robot[i].getField("rotation").getSFRotation())] + target_pos + other_pos)
	# 	# concatenated = np.concatenate([np.array(self.robot[i].getPosition())] + target_pos + other_pos)

	# 	own_pos = []
	# 	own_temp = []
	# 	own_temp.append(np.array(self.robot[i].getPosition())[0])
	# 	own_temp.append(np.array(self.robot[i].getPosition())[2])
	# 	own_pos.append(np.array(own_temp))
	# 	concatenated = np.concatenate(own_pos + target_pos + other_pos)
	# 	return concatenated

	def get_observation_agent(self, i):
		target = np.array(self.box[i].getField("translation").getSFVec3f())
		target[1] = 2.0
		target_pos = []
		target_pos.append(target - np.array(self.robot[i].getField("translation").getSFVec3f()))

		other_pos = []
		for j in range(num_agents):
			if j == i:
				continue
			other_pos.append(np.array(self.robot[j].getField("translation").getSFVec3f()) - 
													np.array(self.robot[i].getField("translation").getSFVec3f()))

		concatenated = np.concatenate([np.array(self.robot[i].getField("translation").getSFVec3f())] + target_pos + other_pos)
		# print(concatenated)
		return concatenated

	def get_reward_agent(self, i, action=None):
		# dist = self.robot[i].getField("translation").getSFVec3f()[2]
		# return -dist

		rew = 0
		one = np.array(self.box[i].getField("translation").getSFVec3f())
		one[1] = 2.0
		two = np.array(self.robot[i].getField("translation").getSFVec3f())
		diff = np.sqrt(np.sum(np.square(one - two)))

		#
		if diff < 1:
			rew += 20
		else:
			rew -= (diff)
		return rew
		#

		# reward = 1-(0.3*(abs(one - two))).sum()
		# return reward

	def is_done_agent(self, i):
		return False

	def get_info_agent(self, i):
		return {}

	##################
	##################
	##################
	##################

	def solved(self):
		# if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
		# 	if np.mean(self.episodeScoreList[-100:]) > 195.0:  # Last 100 episodes' scores average value
		# 		return True
		return False

	def reset(self):
		self.respawnRobot()
		self.supervisor.simulationResetPhysics()  # Reset the simulation physics to start over
		# time.sleep(5)
		
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
			# actind_i.append(ind)
			# actind_i.append(i)

			# continuous
			for j in action[0][i]:
				actind_i.append(j)
			actind_i.append(i)

		self.handle_emitter(actind_i)

	def stepup(self):
		for i in range(num_agents):
			one = np.array(self.box[i].getField("translation").getSFVec3f())
			one[1] = 2.0
			two = np.array(self.robot[i].getField("translation").getSFVec3f())
			diff = np.sqrt(np.sum(np.square(one - two)))


			if diff < 1:
				print('picked')
				box_node = self.supervisor.getFromDef("LOC"+str(i+1))
				trans_field = box_node.getField("translation")
				pos = np.random.uniform(0, 5, 3)
				pos[1] = 0.001

				location = pos.tolist()
				trans_field.setSFVec3f(location)
				box_node.resetPhysics()

	def step(self, action):
		if self.supervisor.step(self.timestep) == -1:
			exit()

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

		self.handle_emitter(actind_i)
		obs_n = []
		for i in range(num_agents):
			obs_n.append(self.get_observation_agent(i))
		single_env_obs = []
		single_env_obs.append(obs_n)
		rews_n = []
		for i in range(num_agents):
			rews_n.append(self.get_reward_agent(i, action[0][i]))
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
cont = True

if train:

	env_id = 'uav'
	model_name = 'maddpg'
	seed = 1
	n_training_threads = 1
	agent_alg = 'MADDPG'
	adversary_alg = 'MADDPG'
	tau = 0.01
	lr = 0.01
	hidden_dim = 32 #64
	buffer_length = int(1e6) #int(1e6)
	n_rollout_threads = 1
	n_exploration_eps = 25000
	final_noise_scale = 0.0
	init_noise_scale = 0.3
	batch_size = 2048 #1024
	steps_per_update = 100
	save_interval = 500

	n_episodes = 10000
	episode_length = 1420 #1920

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
	# run_dir = model_dir / curr_run
	# log_dir = run_dir / 'logs'
	# os.makedirs(log_dir)
	logger = SummaryWriter(str(log_dir))

	torch.manual_seed(seed)
	np.random.seed(seed)
	if not USE_CUDA:
		torch.set_num_threads(n_training_threads)

	supervisor = CartPoleSupervisor()
	supervisor.load_locations()

	if is_new_file == True:
		maddpg = MADDPG.init_from_env(supervisor, agent_alg=agent_alg, adversary_alg=adversary_alg, tau=tau, lr=lr, hidden_dim=hidden_dim)
	else:
		model_path = run_dir / 'model.pt'
		maddpg = MADDPG.init_from_save(model_path)

	obsp, acsp = [], []
	for i in range(num_agents):
		obsp.append(9)
		acsp.append(2)

	replay_buffer = ReplayBuffer(buffer_length, supervisor.num_agents, obsp, acsp)
	t = 0
	file.seek(0)
	index_val = int(file.read())

	file_t.seek(0)
	index_val_t = int(file_t.read())
	# print('index_val', index_val)
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

		temp_act_init = []
		for i in range(num_agents):
			temp_act_init.append(np.array([supervisor.robot[i].getField("translation").getSFVec3f()[2], supervisor.robot[i].getField("translation").getSFVec3f()[0]]))
		temp_actions = []
		temp_actions.append(temp_act_init)

		for et_i in range(check_point):
			# supervisor.flight(et_i)
			supervisor.step_init(temp_actions)
			itm_et_i = et_i
		for et_i in range(itm_et_i):
			if supervisor.check():
				break
			torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(maddpg.nagents)]
			torch_agent_actions = maddpg.step(torch_obs, explore=True)
			agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
			actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]
			# print('actions', actions)
			global_et_i = et_i+1
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
	# curr_agent = maddpg.agents[0]
	maddpg.prep_rollouts(device='cpu')

	for ep_i in range(n_episodes):
		print("Episode %i of %i" % (ep_i + 1, n_episodes))
		obs = supervisor.reset()

		check_point = 400
		itm_et_i = 0
		
		temp_act_init = []
		for i in range(num_agents):
			temp_act_init.append(np.array([supervisor.robot[i].getField("translation").getSFVec3f()[2], supervisor.robot[i].getField("translation").getSFVec3f()[0]]))
		temp_actions = []
		temp_actions.append(temp_act_init)

		for t_i in range(check_point):
			supervisor.step_init(temp_actions)
			itm_et_i = t_i
		for t_i in range(itm_et_i):
			if supervisor.check():
				break
			# supervisor.flight(t_i)
			# rearrange observations to be per agent, and convert to torch Variable
			torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(maddpg.nagents)]
			# torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1), requires_grad=False) for i in range(maddpg.nagents)]

			# # get actions as torch Variables
			# torch_actions = maddpg.step(torch_obs, explore=False)
			# # convert actions to numpy arrays
			# actions = [ac.data.numpy().flatten() for ac in torch_actions]
			# obs, rewards, dones, infos = supervisor.step(actions)

			# print(torch_obs)
			# value = curr_agent.policy(torch_obs[0])
			# tar_value = curr_agent.target_policy(torch_obs[0])
			# print('value', value)
			torch_agent_actions = maddpg.step(torch_obs, explore=True)
			agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
			actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]
			# print('actions eval', actions)
			for i in range(num_agents):
				actions[0][i][0] = 2 * actions[0][i][0] + supervisor.robot[i].getField("translation").getSFVec3f()[2]
				actions[0][i][1] = 2 * actions[0][i][1] + supervisor.robot[i].getField("translation").getSFVec3f()[0]
			next_obs, rewards, dones, infos = supervisor.step(actions)

	supervisor.close()
