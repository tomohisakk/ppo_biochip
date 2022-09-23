import ptan
import ptan.ignite as ptan_ignite
import gym
import argparse
import random
import time
import torch as T
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Engine
from types import SimpleNamespace
from lib import common, ppo

from sub_envs.static import MEDAEnv

class Params():
	lr = 1e-4
	entropy_beta = 1e-3
	batch_size = 32
	ppo_epoches = 8
#	sgamma = 0.1

	w = 8
	h = 8
	dsize = 1
	p = 1.0
	useGPU = True

	env_name = "test"
	gamma = 0.99
	gae_lambda = 0.95
	ppo_eps =  0.1
	ppo_trajectory = 1025
	stop_test_reward = 10000
	stop_reward = None

params = Params()

if __name__ == "__main__":
	env = MEDAEnv(w=params.w, h=params.h, dsize=params.dsize, p=params.p)

	if params.useGPU == True:
		device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
	else:
		device = T.device('cpu')
	print("Device is ", device)

	net = ppo.AtariBasePPO(env.observation_space, env.action_space).to(device)
	print(net)

	@T.no_grad()
	def get_distill_reward(obs) -> float:
		obs_t = T.FloatTensor([obs]).to(device)
		res = (dist_ref(obs_t) - dist_trn(obs_t)).abs()[0][0].item()
		return res


	agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor,
								device=device)

	exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

	optimizer = optim.SGD(net.parameters(), lr=params.lr, momentum=0.9)
#	optimizer = optim.Adam(net.parameters(), lr=params.lr)
#	scheduler = T.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.sgamma)

	def process_batch(engine, batch):
		start_ts = time.time()
		optimizer.zero_grad()
		res = {}

		states_t, actions_t, adv_t, ref_t, old_logprob_t = batch
		policy_t, value_t = net(states_t)
		loss_value_t = F.mse_loss(value_t.squeeze(-1), ref_t)
		res['ref'] = ref_t.mean().item()

		logpolicy_t = F.log_softmax(policy_t, dim=1)

		prob_t = F.softmax(policy_t, dim=1)
		loss_entropy_t = (prob_t * logpolicy_t).sum(dim=1).mean()

		logprob_t = logpolicy_t.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
		ratio_t = T.exp(logprob_t - old_logprob_t)
		surr_obj_t = adv_t * ratio_t
		clipped_surr_t = adv_t * T.clamp(ratio_t, 1.0 - params.ppo_eps, 1.0 + params.ppo_eps)
		loss_policy_t = -T.min(surr_obj_t, clipped_surr_t).mean()

		loss_t = params.entropy_beta * loss_entropy_t + loss_policy_t + loss_value_t
		loss_t.backward()
		optimizer.step()


		res.update({
			"loss": loss_t.item(),
			"loss_value": loss_value_t.item(),
			"loss_policy": loss_policy_t.item(),
			"adv": adv_t.mean().item(),
			"loss_entropy": loss_entropy_t.item(),
			"time_batch": time.time() - start_ts,
		})

		return res


	engine = Engine(process_batch)

	common.setup_ignite(engine, params, exp_source, params.env_name, extra_metrics=(
		'test_reward', 'avg_test_reward', 'test_steps'))

	@engine.on(ptan_ignite.PeriodEvents.ITERS_100000_COMPLETED)
	def test_network(engine):
		net.actor.train(False)
		obs = env.reset()
		reward = 0.0
		steps = 0

		while True:
			acts, _ = agent([obs])
			obs, r, is_done, _ = env.step(acts[0])
			reward += r
			steps += 1
			if is_done:
				break
		test_reward_avg = getattr(engine.state, "test_reward_avg", None)
		if test_reward_avg is None:
			test_reward_avg = reward
		else:
			test_reward_avg = test_reward_avg * 0.95 + 0.05 * reward
		engine.state.test_reward_avg = test_reward_avg
		print("Test done: got %.3f reward after %d steps, avg reward %.3f" % (
			reward, steps, test_reward_avg
		))
		engine.state.metrics['test_reward'] = reward
		engine.state.metrics['avg_test_reward'] = test_reward_avg
		engine.state.metrics['test_steps'] = steps

		if test_reward_avg > params.stop_test_reward:
			print("Reward boundary has crossed, stopping training. Contgrats!")
			engine.should_terminate = True
#		net.actor.train(True)

#		scheduler.step()

	def new_ppo_batch():
		# In noisy networks we need to reset the noise
		pass

	engine.run(ppo.batch_generator(exp_source, net, params.ppo_trajectory,
									params.ppo_epoches, params.batch_size,
									params.gamma, params.gae_lambda, device=device,
									trim_trajectory=False, new_batch_callable=new_ppo_batch))
