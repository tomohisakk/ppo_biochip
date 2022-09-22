import copy
import queue
import random
import numpy as np
from enum import IntEnum
import math
import collections
import gym

from sub_envs.map import MakeMap
from sub_envs.map import Symbols

class Actions(IntEnum):
	N = 0
	E = 1
	S = 2
	W = 3

class MEDAEnv(gym.Env):
	def __init__(self, w=8, h=8, dsize=2, p=0.8, test_flag=False):
		super(MEDAEnv, self).__init__()
		assert w>0 and h>0 and dsize>0
		assert 0<=p<=1.0
		self.w = w
		self.h = h
		self.dsize = dsize
		self.p = p
		self.actions = Actions
		self.action_space = len(self.actions)
		self.observation_space = (w, h, 3)
		self.n_steps = 0
		self.max_step = 2*(w+h)

		self.state = (0,0)
		self.goal = (w-1, h-1)

		self.map_symbols = Symbols()
		self.mapclass = MakeMap(w=self.w,h=self.h,dsize=self.dsize,p=self.p)
		self.map = self.mapclass.gen_random_map()

		self.test_flag = test_flag

	def reset(self, test_map=None):
		self.n_steps = 0
		self.state = (0, 0)

		if self.test_flag == False:
			self.map = self.mapclass.gen_random_map()
		else:
			self.map = test_map

		obs = self._get_obs()

		return obs

	def step(self, action):
		done = False
		self.n_steps += 1
		message = None

		_dist = self._get_dist(self.state, self.goal)
#		print(_dist)
		self._update_position(action)
		dist = self._get_dist(self.state, self.goal)
#		print(self.map)

		if dist <= (self.dsize-1)*math.sqrt(2):
			reward = 1.0
			done = True
#			print("okok1")
		elif self.n_steps == self.max_step:
			reward = -0.8
			done = True
		elif dist < _dist:
			reward = 0.5
		elif dist == _dist:
			reward = -0.5
		else:
			reward = -0.8

#		if self.test_flag == True:
#			print()
#		print(self.map)

		obs = self._get_obs()
#		print(obs)
		return obs, reward, done, message

	def _get_dist(self, state1, state2):
		diff_x = state1[1] - state2[1]
		diff_y = state1[0] - state2[0]
		return math.sqrt(diff_x*diff_x + diff_y*diff_y)

	def _is_touching(self, dstate, obj):
		i = 0
		while True:
			j = 0
			while True:
				if self.map[dstate[1]+j][dstate[0]+i] == obj:
					return True
				j += 1
				if j == self.dsize:
					break
			i += 1
			if i == self.dsize:
				break

		return False

	def _update_position(self, action):
		state_ = list(self.state)

		if action == Actions.N:
			state_[1] -= 1
		elif action == Actions.E:
			state_[0] += 1
		elif action == Actions.S:
			state_[1] += 1
		else:
			state_[0] -= 1

		if (0 <= state_[1] < self.h-self.dsize+1) and (0 <= state_[0] < self.w-self.dsize+1) and\
		   (self._is_touching(state_, self.map_symbols.Dynamic_module) == False) and (self._is_touching(state_, self.map_symbols.Static_module) == False):
			i = 0
			while True:
				j = 0
				while True:
					self.map[self.state[1]+j][self.state[0]+i] = self.map_symbols.Health
					j += 1
					if j == self.dsize:
						break
				i += 1
				if i == self.dsize:
					break

			self.state = state_

			# Set Droplet state
			i = 0
			while True:
				j = 0
				while True:
					self.map[self.state[1]+j][self.state[0]+i] = self.map_symbols.State
					j += 1
					if j == self.dsize:
						break
				i += 1
				if i == self.dsize:
					break


	def _get_obs(self):
		obs = np.zeros(shape = (self.w, self.h, 3))
		for i in range(self.w):
			for j in range(self.h):
				if self.map[j][i] == self.map_symbols.State:
					obs[i][j][0] = 1
				elif self.map[j][i] == self.map_symbols.Goal:
					obs[i][j][1] = 1
				elif self.map[j][i] == self.map_symbols.Dynamic_module or self.map[j][i] == self.map_symbols.Static_module:
					obs[i][j][2] = 1
#		print(obs)
		return obs

	def close(self):
		pass

"""
if __name__ == '__main__':
	env = MEDAEnv(w=8, h=8, dsize=2, p=0.9)
	done = False
	while not done:
		action = np.random.randint(0,4)
		print(action)
		obs, reward, done, _ = env.step(action)
		#env.reset(n_modules=5)
		#print("action and obs")
#		print(action)
#		print(obs)
#		print(reward)
	print("--------------SECOND RAUND----------")
	env.reset()
	done = 0
	while not done:
		action = np.random.randint(0,4)
		print(action)
		obs, reward, done, _ = env.step(action)
		#env.reset(n_modules=5)
		#print("action and obs")
#		print(action)
#		print(obs)
#		print(reward)
"""