import random
import pickle
import torch as T
import numpy as np
import collections

#from sub_envs.static import MEDAEnv
from sub_envs.dynamic import MEDAEnv
from sub_envs.map import MakeMap
from sub_envs.map import Symbols
from lib import common, ppo

def _is_touching(dstate, obj, map, dsize):
		i = 0
		while True:
			j = 0
			while True:
				if map[dstate[1]+j][dstate[0]+i] == obj:
					return True
				j += 1
				if j == dsize:
					break
			i += 1
			if i == dsize:
				break

		return False

def _compute_shortest_route(w, h, dsize, symbols,map, start):
	queue = collections.deque([[start]])
	seen = set([start])
#		print(self.map)
	while queue:
		path = queue.popleft()
#			print(path)
		x, y = path[-1]
		if _is_touching((x,y), symbols.Goal, map, dsize):
			return path
		for x2, y2 in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
			if 0 <= x2 < (w-dsize+1) and 0 <= y2 < (h-dsize+1) and \
			(_is_touching((x2,y2), symbols.Dynamic_module, map, dsize) == False) and\
			(_is_touching((x2,y2), symbols.Static_module, map, dsize) == False) and\
			(x2, y2) not in seen:
				queue.append(path + [(x2, y2)])
				seen.add((x2, y2))
#		print("Bad map")
#		print(self.map)
	return False

if __name__ == "__main__":
	###### Set params ##########

	W = 8
	H = 8
	DSIZE = 1
	S_MODULES = 0
	D_MODULES = 0
	N_EPOCH = 1

	############################

	random.seed(123)
	T.manual_seed(123)

	IS_IMPORT = True
	ENV_NAME = str(W)+str(H)+str(DSIZE)+str(S_MODULES)+str(D_MODULES) + "/" + str(N_EPOCH)
	env = MEDAEnv(w=W, h=H, dsize=DSIZE, s_modules=S_MODULES, d_modules=D_MODULES, test_flag=True)

	if IS_IMPORT:
		dir_name = "testmaps/%sx%s/%s/%s,%s"%(W , H, DSIZE, S_MODULES, D_MODULES)
		file_name = "%s/map.pkl"%(dir_name)

		save_file = open(file_name, "rb")
		maps = pickle.load(save_file)

	device = T.device('cpu')

	CHECKPOINT_PATH = "saves/" + ENV_NAME

	net = ppo.PPO(env.observation_space, env.action_space).to(device)
	net.load_checkpoint(ENV_NAME)
	net.eval()

	n_games = 0

	n_critical = 0

	map_symbols = Symbols()
	mapclass = MakeMap(w=W,h=H,dsize=DSIZE,s_modules=S_MODULES, d_modules=D_MODULES)

	for n_games in range(10):
		if IS_IMPORT:
			map = maps[n_games]
		else:
			map = mapclass.gen_random_map()
#		print(map)

		observation = env.reset(test_map=map)

		done = False
		score = 0
		n_steps = 0

		path = _compute_shortest_route(W, H, DSIZE, map_symbols, map, (0,0))

		while not done:
			observation = T.tensor([observation], dtype=T.float)
			acts, _ = net(observation)
			action = T.argmax(acts).item()
			observation, reward, done, message = env.step(action)
			score += reward

			if message == None:
				n_steps += 1

			if done:
				break

		if len(path)-1 == n_steps:
			n_critical += 1

print("Critical path: ", n_critical)
print("Avg of critical path: ", n_critical/10)

if IS_IMPORT:
	save_file.close()
