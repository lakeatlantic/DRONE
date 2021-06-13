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

	def respawnRobot(self, uav, tar):
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
			trans_field = robot_node.getField("translation")
			pos = np.random.uniform(-1, 1, 3)
			pos[0] = uav[0]
			pos[1] = 0.1
			pos[2] = uav[1]

			location = pos.tolist()
			trans_field.setSFVec3f(location)
			robot_node.resetPhysics()

		# box respawn
		for i in range(num_agents):
			self.box.append(self.supervisor.getFromDef("LOC"+str(i+1)))
			box_node = self.supervisor.getFromDef("LOC"+str(i+1))
			trans_field = box_node.getField("translation")
			pos = np.random.uniform(-1, 1, 3)
			pos[0] = tar[0]
			pos[1] = 0.0001
			pos[2] = tar[1]

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

	def reset(self, uav, tar):
		self.respawnRobot(uav, tar)
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

		# p_force = [None] * (self.num_agents)
		# # print('1', p_force)
		# for i,agent in enumerate(self.robot):
		# 	noise = np.random.randn(2) * self.agent_u_noise if self.agent_u_noise else 0.0
		# 	p_force[i] = (self.agent_mass * self.agent_accel if self.agent_accel is not None else 
		# 		self.agent_mass) * action[0][i] + noise

		# # print('2', p_force)
		# for i,agent in enumerate(self.robot):
		# 	self.agent_p_vel = self.agent_p_vel * (1 - self.damping)
		# 	if (p_force[i] is not None):
		# 		self.agent_p_vel += (p_force[i] / self.agent_mass) * self.dt
		# 	if self.max_speed is not None:
		# 		speed = np.sqrt(np.square(self.agent_p_vel[0]) + np.square(self.agent_p_vel[1]))
		# 		if speed > self.max_speed:
		# 			self.agent_p_vel = self.agent_p_vel / np.sqrt(np.square(self.agent_p_vel[0]) +
		# 														  np.square(self.agent_p_vel[1])) * self.max_speed
		# 	action[0][i][0] = self.robot[i].getField("translation").getSFVec3f()[2] + self.agent_p_vel[0] * self.dt
		# 	action[0][i][1] = self.robot[i].getField("translation").getSFVec3f()[0] + self.agent_p_vel[1] * self.dt

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

# train = True
train = False
# that_one = True
that_one = False

waypts = [[2,1,2], [1,1,1], [0,1,0], [-1,1,-1], [-2,1,-2]]
way_ind = 1
cur_tar = waypts[way_ind]

if train:

	env_id = 'uav'
	model_name = 'maddpg'
	seed = 1
	n_rollout_threads = 1
	n_training_threads = 1
	buffer_length = int(1e6)
	n_episodes = 3000
	steps_per_update = 100
	batch_size = 1024
	n_exploration_eps = 25000
	init_noise_scale = 0.3
	final_noise_scale = 0.0
	save_interval = 500
	hidden_dim = 64
	lr = 0.01
	tau = 0.01
	agent_alg = 'MADDPG'
	adversary_alg = 'MADDPG'
	
	if that_one:
		episode_length = 150 #1920
	else:
		episode_length = 150 #1920

	USE_CUDA = False

	model_dir = Path('/home/sagar/models') / env_id / model_name

	if not model_dir.exists():
		curr_run = 'run1'
	else:
		exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
						 model_dir.iterdir() if
						 str(folder.name).startswith('run')]
		if len(exst_run_nums) == 0:
			curr_run = 'run1'
		else:
			curr_run = 'run%i' % (max(exst_run_nums) + 1)
	run_dir = model_dir / curr_run
	log_dir = run_dir / 'logs'
	os.makedirs(log_dir)
	logger = SummaryWriter(str(log_dir))

	torch.manual_seed(seed)
	np.random.seed(seed)
	if not USE_CUDA:
		torch.set_num_threads(n_training_threads)
	env = CartPoleSupervisor()
	maddpg = MADDPG.init_from_env(env, agent_alg=agent_alg,
								  adversary_alg=adversary_alg,
								  tau=tau,
								  lr=lr,
								  hidden_dim=hidden_dim)
	obsp, acsp = [], []
	for i in range(num_agents):
		obsp.append(4)
		acsp.append(2)

	replay_buffer = ReplayBuffer(buffer_length, env.num_agents, obsp, acsp)

	t = 0
	for ep_i in range(0, n_episodes, n_rollout_threads):
		print("Episodes %i-%i of %i" % (ep_i + 1,
										ep_i + 1 + n_rollout_threads,
										n_episodes))
		obs = env.reset()
		env.agent_p_vel = np.array([0.0, 0.0])
		# obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
		maddpg.prep_rollouts(device='cpu')

		explr_pct_remaining = max(0, n_exploration_eps - ep_i) / n_exploration_eps
		maddpg.scale_noise(final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)
		maddpg.reset_noise()

		# init actions
		temp_act_init = []
		for i in range(num_agents):
			if that_one:
				temp_act_init.append(np.array([0.0, 0.0]))
			else:
				temp_act_init.append(np.array([env.robot[i].getField("translation").getSFVec3f()[2], 
					env.robot[i].getField("translation").getSFVec3f()[0]]))
		temp_actions = []
		temp_actions.append(temp_act_init)

		# rise to given altitude
		for et_i in range(100):
			env.step_init(temp_actions)

		for et_i in range(episode_length):
			# rearrange observations to be per agent, and convert to torch Variable
			torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
								  requires_grad=False)
						 for i in range(maddpg.nagents)]
			# get actions as torch Variables
			torch_agent_actions = maddpg.step(torch_obs, explore=True)
			# convert actions to numpy arrays
			agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
			# rearrange actions to be per environment
			actions = [[ac[i] for ac in agent_actions] for i in range(n_rollout_threads)]

			# if that_one:
			# 	pass
			# else:
			# 	for i in range(num_agents):
			# 		actions[0][i][0] = 2 * actions[0][i][0] + env.robot[i].getField("translation").getSFVec3f()[2]
			# 		actions[0][i][1] = 2 * actions[0][i][1] + env.robot[i].getField("translation").getSFVec3f()[0]

			next_obs, rewards, dones, infos = env.step(actions)
			replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
			obs = next_obs
			t += n_rollout_threads
			if (len(replay_buffer) >= batch_size and
				(t % steps_per_update) < n_rollout_threads):
				if USE_CUDA:
					maddpg.prep_training(device='gpu')
				else:
					maddpg.prep_training(device='cpu')
				for u_i in range(n_rollout_threads):
					for a_i in range(maddpg.nagents):
						sample = replay_buffer.sample(batch_size,
													  to_gpu=USE_CUDA)
						maddpg.update(sample, a_i, logger=logger)
					maddpg.update_all_targets()
				maddpg.prep_rollouts(device='cpu')
		ep_rews = replay_buffer.get_average_rewards(
			episode_length * n_rollout_threads)
		for a_i, a_ep_rew in enumerate(ep_rews):
			logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

		if ep_i % save_interval < n_rollout_threads:
			os.makedirs(run_dir / 'incremental', exist_ok=True)
			maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
			maddpg.save(run_dir / 'model.pt')

	maddpg.save(run_dir / 'model.pt')
	env.close()
	logger.export_scalars_to_json(str(log_dir / 'summary.json'))
	logger.close()

else:

	env_id = 'uav'
	model_name = 'maddpg'
	run_num = 1
	save_gifs = False
	incremental = None
	n_rollout_threads = 1
	n_episodes = 10
	episode_length = 150

	model_path = (Path('/home/sagar/models') / env_id / model_name /
				  ('run%i' % run_num))
	if incremental is not None:
		model_path = model_path / 'incremental' / ('model_ep%i.pt' %
												   incremental)
	else:
		model_path = model_path / 'model.pt'

	maddpg = MADDPG.init_from_save(model_path)
	env = make_env(env_id, discrete_action=maddpg.discrete_action)
	e = CartPoleSupervisor()
	maddpg.prep_rollouts(device='cpu')

	for ep_i in range(n_episodes):
		print("Episode %i of %i" % (ep_i + 1, n_episodes))
		obs = env.reset()
		uav = np.array([0.0, 0.0])
		tar = np.array([0.0, 0.0])
		uav[0], uav[1], tar[0], tar[1] = obs[0][0][0], obs[0][0][1], obs[0][0][2], obs[0][0][3]
		obs_e = e.reset(uav, tar)

		for t_i in range(episode_length):
			# rearrange observations to be per agent, and convert to torch Variable
			torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
								  requires_grad=False)
						 for i in range(maddpg.nagents)]
			# get actions as torch Variables
			torch_actions = maddpg.step(torch_obs, explore=False)
			# convert actions to numpy arrays
			actions = [ac.data.numpy().flatten() for ac in torch_actions]
			obs, rewards, dones, infos = env.step(actions)
			uav[0], uav[1], tar[0], tar[1] = obs[0][0][0], obs[0][0][1], obs[0][0][2], obs[0][0][3]
			act = np.array([uav[0], uav[1]])
			actt = []
			actt.append(act)
			obs, rewards, dones, infos = e.step(actt)

	env.close()
