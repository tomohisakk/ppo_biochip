import gym
import collections
import numpy as np
import warnings
import torch
import torch.nn as nn
from datetime import timedelta, datetime
from types import SimpleNamespace
from typing import Iterable, Tuple, List

import ptan
import ptan.ignite as ptan_ignite
from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

GAMES = 30000
EPOCHS = 200

def setup_ignite(engine: Engine, params: SimpleNamespace,
				exp_source, run_name: str, net, optimizer, scheduler,
				extra_metrics: Iterable[str] = ()):
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

		if (trainer.state.episode%GAMES == 0)and(optimizer.param_groups[0]['lr'] > 1e-6):
			scheduler.step()
			print("=== Current LR ===")
			print(optimizer.param_groups[0]['lr'])

		if trainer.state.episode % (2*GAMES) == 0:
			net.save_checkpoint(params.env_name)

		if trainer.state.episode % 1000 == 0:
			mean_reward = np.mean(total_rewards[-GAMES:])
			mean_n_steps = np.mean(total_n_steps_ep[-GAMES:])
			passed = trainer.state.metrics.get('time_passed', 0)
			print("Episode/Games %d/%d: reward=%.2f, steps=%d, "
				"speed=%.1f f/s, elapsed=%s" % (
				trainer.state.episode/GAMES, trainer.state.episode, 
				mean_reward, mean_n_steps,
				trainer.state.metrics.get('avg_fps', 0),
				timedelta(seconds=int(passed))))

		if trainer.state.episode == GAMES*EPOCHS:
			engine.terminate()
			print("=== Learning end ===")


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

	# write to tensorboard every 100 iterations
	ptan_ignite.PeriodicEvents().attach(engine)
	metrics = ['avg_loss', 'avg_fps']
	metrics.extend(extra_metrics)
	handler = tb_logger.OutputHandler(
		tag="train", metric_names=metrics,
		output_transform=lambda a: a)
	event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
	tb.attach(engine, log_handler=handler, event_name=event)
