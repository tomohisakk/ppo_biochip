import pickle
import torch as T
import numpy as np
import collections

from sub_envs.static import MEDAEnv
#from sub_envs.dynamic import MEDAEnv
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
	ENV_NAME = "s8813"
	TOTAL_GAMES = 10000

	W = 8
	H = 8
	DSIZE = 1
	N_MODULES = 3

	IS_IMPORT = False
	lmaps = []
	############################
	env = MEDAEnv(w=W, h=H, dsize=DSIZE, n_modules=N_MODULES, test_flag=True)

	if IS_IMPORT:
		dir_name = "testmaps/size:%sx%s/dsize:%s/modlue:%s"%(W , H, DSIZE, N_MODULES)
		file_name = "%s/map.pkl"%(dir_name)

		save_file = open(file_name, "rb")
		maps = pickle.load(save_file)

	device = T.device('cpu')

	CHECKPOINT_PATH = "saves/" + ENV_NAME

	net = ppo.PPO(env.observation_space, env.action_space).to(device)
	net.load_checkpoint(ENV_NAME)

	n_games = 0

	n_critical = 0

	map_symbols = Symbols()
	mapclass = MakeMap(w=W,h=H,dsize=DSIZE,n_modules=N_MODULES)

	for n_games in range(TOTAL_GAMES):
		if IS_IMPORT:
			map = maps[n_games]
		else:
			map = mapclass.gen_random_map()
#		print(map)

		net.actor.train(False)
		observation = env.reset(test_map=map)

		done = False
		score = 0
		n_steps = 0

		path = _compute_shortest_route(W, H, DSIZE, map_symbols, map, (0,0))

		while not done:
			observation = T.tensor([observation], dtype=T.float)
			acts, _ = net(observation)
#			print(acts)
			action = T.argmax(acts).item()
			observation, reward, done, message = env.step(action)
			score += reward

			if message == None:
				n_steps += 1

			if done:
				break

#		print("shortest:",len(path))
#		print("stepnum:",n_steps)

		if len(path)-1 == n_steps:
			n_critical += 1
#		else:
#			print()
#			print(observation[2])
#			print(n_steps)

#		writer.add_scalar("Step_num", n_steps, n_games)

print("Finish " + str(TOTAL_GAMES) + " tests")
print("Critical path: ", n_critical)
print("Avg of critical path: ", n_critical/TOTAL_GAMES)

if IS_IMPORT:
	save_file.close()
