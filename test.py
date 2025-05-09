import torch
from time import time
from env import Env_tsp
from config import Config, load_pkl, pkl_parser
from search import sampling, active_search
	
def search_tour(cfg, env):
	test_input = env.get_nodes(cfg.seed)  # 随机生成20个城市节点
	
	# random 
	print('generate random tour...')
	random_tour = env.get_random_tour()  # 随机生成一条路径作为基准路径
	env.show(test_input, random_tour)  # 路径可视化
	
	# simplest way
	print('sampling ...')
	t1 = time()
	pred_tour = sampling(cfg, env, test_input)  # 使用Actor模型生成路径
	t2 = time()
	print('%dmin %1.2fsec\n'%((t2-t1)//60, (t2-t1)%60))
	env.show(test_input, pred_tour)
	
	# active search, update parameters during test
	print('active search ...')
	t1 = time()
	pred_tour = active_search(cfg, env, test_input)  # 优化生成的路径
	t2 = time()
	print('%dmin %1.2fsec\n'%((t2-t1)//60, (t2-t1)%60))
	env.show(test_input, pred_tour)
	
	"""
	# optimal solution, it takes time
	print('generate optimal solution ...')
	t1 = time()
	optimal_tour = env.get_optimal_tour(test_input)
	env.show(test_input, optimal_tour)
	t2 = time()
	print('%dmin %1.2fsec\n'%((t2-t1)//60, (t2-t1)%60))
	"""
	
if __name__ == '__main__':
	cfg = load_pkl(pkl_parser().path)
	env = Env_tsp(cfg)
		
	# inputs = env.stack_nodes()
	# ~ tours = env.stack_random_tours()
	# ~ l = env.stack_l(inputs, tours)
	
	# ~ nodes = env.get_nodes(cfg.seed)
	# random_tour = env.get_random_tour()
	# ~ env.show(nodes, random_tour)
	
	# ~ env.show(inputs[0], random_tour)
	# ~ inputs = env.shuffle_index(inputs)
	# env.show(inputs[0], random_tour)

	# inputs = env.stack_nodes()
	# random_tour = env.get_random_tour()
	# env.show(inputs[0], random_tour)

		
	if cfg.mode == 'test':
		search_tour(cfg, env)

	else:
		raise NotImplementedError('test only, specify test pkl file')
		
