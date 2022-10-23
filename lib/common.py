import ptan
import warnings
import torch as T
T.manual_seed(0)
import numpy as np
import torch.nn as nn
from typing import Iterable
from ignite.engine import Engine
import ptan.ignite as ptan_ignite
from types import SimpleNamespace
from datetime import timedelta, datetime
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

import pickle
import numpy as np
import collections
#from sub_envs.static import MEDAEnv
from sub_envs.dynamic import MEDAEnv
from sub_envs.map import MakeMap
from sub_envs.map import Symbols
from lib import ppo

GAMES = 30000

def setup_ignite(engine: Engine, params: SimpleNamespace, exp_source, run_name: str, 
				 net, optimizer, scheduler, extra_metrics: Iterable[str] = ()):
	warnings.simplefilter("ignore", category=UserWarning)
	handler = ptan_ignite.EndOfEpisodeHandler(exp_source, bound_avg_reward=params.stop_reward)
	handler.attach(engine)
	ptan_ignite.EpisodeFPSHandler().attach(engine)

	total_rewards = []
	total_n_steps_ep = []

	@engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
	def episode_completed(trainer: Engine):
		total_rewards.append(trainer.state.episode_reward)
		total_n_steps_ep.append(trainer.state.episode_steps)

		if trainer.state.episode % 3000 == 0:
			mean_reward = np.mean(total_rewards[-GAMES:])
			mean_n_steps = np.mean(total_n_steps_ep[-GAMES:])
			passed = trainer.state.metrics.get('time_passed', 0)
			print("Episode/Games %d/%d: reward=%.2f, steps=%d, "
				"speed=%.1f f/s, elapsed=%s" % (
				trainer.state.episode/GAMES, trainer.state.episode, 
				mean_reward, mean_n_steps,
				trainer.state.metrics.get('avg_fps', 0),
				timedelta(seconds=int(passed))))

		if trainer.state.episode%GAMES == 0:
#			if (optimizer.param_groups[0]['lr'] > 1e-6):
#				scheduler.step()
#				print("LR: ", optimizer.param_groups[0]['lr'])
#			else:
			save_name = params.env_name + "/" +str(int(trainer.state.episode/GAMES))
			net.save_checkpoint(save_name)
			if test(save_name, params.w, params.h, params.dsize, params.s_modules, params.d_modules) == 1.0:
				engine.terminate()
				print("=== Learning end ===")
#				critical_ctr += 1
#				print(save_name + " is critical")
#				print("critical_ctr:", critical_ctr)
#				if critical_ctr == 5:

	now = datetime.now().isoformat(timespec='minutes')
	logdir = f"runs/{now}-{params.env_name}"
	tb = tb_logger.TensorboardLogger(log_dir=logdir)
	run_avg = RunningAverage(output_transform=lambda v: v['loss'])
	run_avg.attach(engine, "avg_loss")

	metrics = ['reward', 'steps', 'avg_reward']
	handler = tb_logger.OutputHandler(
		tag="episodes", metric_names=metrics)
	event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
	tb.attach(engine, log_handler=handler, event_name=event)

	ptan_ignite.PeriodicEvents().attach(engine)
	metrics = ['avg_loss', 'avg_fps']
	metrics.extend(extra_metrics)
	handler = tb_logger.OutputHandler(
		tag="train", metric_names=metrics,
		output_transform=lambda a: a)
	event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
	tb.attach(engine, log_handler=handler, event_name=event)


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

def test(test_name, w, h, dsize, s_modules, d_modules):
	###### Set params ##########
	############################
	env = MEDAEnv(w, h, dsize, s_modules, d_modules)

	dir_name = "testmaps/%sx%s/%s/%s,%s"%(w , h, dsize, s_modules, d_modules)
	file_name = "%s/map.pkl"%(dir_name)

	save_file = open(file_name, "rb")
	maps = pickle.load(save_file)

	device = T.device('cpu')

	net = ppo.PPO(env.observation_space, env.action_space).to(device)
	net.load_checkpoint(test_name)
	net.eval()

	n_games = 0
	n_critical = 0
	unreach_flag = False

	map_symbols = Symbols()
	mapclass = MakeMap(w,h,dsize,s_modules, d_modules)

	for n_games in range(10000):
		tmap = maps[n_games]

		observation = env.reset(test_map=tmap)

		done = False
		score = 0
		n_steps = 0

		path = _compute_shortest_route(w, h, dsize, map_symbols, tmap, (0,0))

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

#		if 32 > n_steps:
#			n_critical += 1

		if observation[0][w-1][h-1] == 0:
			unreach_flag = True

		if len(path)-1 >= n_steps:
			n_critical += 1
		
	save_file.close()
	print("Test result is ", n_critical/10000)

	if unreach_flag:
		return 0
	else:
		return (n_critical/10000)
