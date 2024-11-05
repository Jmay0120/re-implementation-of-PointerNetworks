import pickle  # 负责序列化（保存）和反序列化（加载）Python对象
import os  # 文件路径和文件夹管理
import argparse  # 命令行参数解析
import torch  # 用于深度学习操作的库
from datetime import datetime  # 处理日期和时间

def argparser():
	parser = argparse.ArgumentParser()
	# main parts
	# 训练或测试模式选择
	parser.add_argument('-m', '--mode', metavar = 'M', type = str, required = True, choices = ['train', 'train_emv', 'test'], help = 'train or train_emv or test')
	# 批量大小
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 512, help = 'batch size, default: 512')
	# 城市或节点数量（时间序列长度）
	parser.add_argument('-t', '--city_t', metavar = 'T', type = int, default = 20, help = 'number of cities(nodes), time sequence, default: 20')
	# 训练步数（即训练轮数）
	parser.add_argument('-s', '--steps', metavar = 'S', type = int, default = 15000, help = 'training steps(epochs), default: 15000')
	
	# details
	# 嵌入层维度大小
	parser.add_argument('-e', '--embed', metavar = 'EM', type = int, default = 128, help = 'embedding size')
	# 隐藏层大小
	parser.add_argument('-hi', '--hidden', metavar = 'HI', type = int, default = 128, help = 'hidden size')
	# 用于增强探索的logits裁剪值
	parser.add_argument('-c', '--clip_logits', metavar = 'C', type = int, default = 10, help = 'improve exploration; clipping logits')
	# Softmax温度（用于改善探索程度）
	parser.add_argument('-st', '--softmax_T', metavar = 'ST', type = float, default = 1.0, help = 'might improve exploration; softmax temperature default 1.0 but 2.0, 2.2 and 1.5 might yield better results')
	# 优化器选择
	parser.add_argument('-o', '--optim', metavar = 'O', type = str, default = 'Adam', help = 'torch optimizer')
	# 初始化权重的最小值
	parser.add_argument('-minv', '--init_min', metavar = 'MINV', type = float, default = -0.08, help = 'initialize weight minimun value -0.08~')
	# 初始化权重的最大值
	parser.add_argument('-maxv', '--init_max', metavar = 'MAXV', type = float, default = 0.08, help = 'initialize weight ~0.08 maximum value')
	# glimpse函数的数量
	parser.add_argument('-ng', '--n_glimpse', metavar = 'NG', type = int, default = 1, help = 'how many glimpse function')
	# critic模型的过程步骤数
	parser.add_argument('-np', '--n_process', metavar = 'NP', type = int, default = 3, help = 'how many process step in critic; at each process step, use glimpse')
	parser.add_argument('-dt', '--decode_type', metavar = 'DT', type = str, default = 'sampling', choices = ['greedy', 'sampling'], help = 'how to choose next city in actor model')
	
	# train, learning rate
	# 初始学习率
	parser.add_argument('--lr', metavar = 'LR', type = float, default = 1e-3, help = 'initial learning rate')
	# 是否启用学习率衰减、默认启用
	parser.add_argument('--is_lr_decay', action = 'store_false', help = 'flag learning rate scheduler default true')
	# 学习率衰减因子
	parser.add_argument('--lr_decay', metavar = 'LRD', type = float, default = 0.96, help = 'learning rate scheduler, decay by a factor of 0.96 ')
	# 每5000步衰减一次学习率
	parser.add_argument('--lr_decay_step', metavar = 'LRDS', type = int, default = 5e3, help = 'learning rate scheduler, decay every 5000 steps')
	
	# inference
	# 加载actor模型的路径
	parser.add_argument('-ap', '--act_model_path', metavar = 'AMP', type = str, help = 'load actor model path')
	# 用于推理的随机种子
	parser.add_argument('--seed', metavar = 'SEED', type = int, default = 1, help = 'random seed number for inference, reproducibility')
	# 在active search中使用的alpha值
	parser.add_argument('-al', '--alpha', metavar = 'ALP', type = float, default = 0.99, help = 'alpha decay in active search')
	
	# path
	# 是否启用csv日志记录，默认启用
	parser.add_argument('--islogger', action = 'store_false', help = 'flag csv logger default true')
	# 是否启用模型保存功能，默认启用
	parser.add_argument('--issaver', action = 'store_false', help = 'flag model saver default true')
	# 日志记录间隔步数
	parser.add_argument('-ls', '--log_step', metavar = 'LOGS', type = int, default = 10, help = 'logger timing')
	# CSV日志文件存储路径
	parser.add_argument('-ld', '--log_dir', metavar = 'LD', type = str, default = './Csv/', help = 'csv logger dir')
	# 模型文件存储路径
	parser.add_argument('-md', '--model_dir', metavar = 'MD', type = str, default = './Pt/', help = 'model save dir')
	# pkl文件存储路径
	parser.add_argument('-pd', '--pkl_dir', metavar = 'PD', type = str, default = './Pkl/', help = 'pkl save dir')
	
	# GPU相关参数
	parser.add_argument('-cd', '--cuda_dv', metavar = 'CD', type = str, default = '0', help = 'os CUDA_VISIBLE_DEVICE, default single GPU')
	args = parser.parse_args()
	return args

# 该类将解析得到的参数保存为对象属性，并创建必要的存储路径
class Config():
	def __init__(self, **kwargs):	
		for k, v in kwargs.items():
			self.__dict__[k] = v
		self.dump_date = datetime.now().strftime('%m%d_%H_%M')  # 当前时间戳
		self.task = '%s%d'%(self.mode, self.city_t)  # 任务名称
		self.pkl_path = self.pkl_dir + '%s.pkl'%(self.task)  # pkl文件路径
		self.n_samples = self.batch * self.steps  # 总样本数
		for x in [self.log_dir, self.model_dir, self.pkl_dir]:
			os.makedirs(x, exist_ok = True)  # 确保目录存在

# 将配置的所有参数打印出来
def print_cfg(cfg):
	print(''.join('%s: %s\n'%item for item in vars(cfg).items()))

# 将配置对象序列化为pkl文件
def dump_pkl(args, verbose = True, override = None):
	cfg = Config(**vars(args))  # 将命令行参数转换为Config对象
	if os.path.exists(cfg.pkl_path):
		override = input(f'found the same name pkl file "{cfg.pkl_path}".\noverride this file? [y/n]:')  # 找到同名文件，是否覆盖？
	with open(cfg.pkl_path, 'wb') as f:
		if override == 'n':
			raise RuntimeError('modify cfg.pkl_path in config.py as you like')  # 否，请修改pkl路径
		pickle.dump(cfg, f)
		print('--- save pickle file in %s ---\n'%cfg.pkl_path)  # 保存pkl文件于
		if verbose:
			print_cfg(cfg)

# 从pkl文件中加载配置对象
def load_pkl(pkl_path, verbose = True):
	if not os.path.isfile(pkl_path):
		raise FileNotFoundError('pkl_path')  # pkl路径不存在
	with open(pkl_path, 'rb') as f:
		cfg = pickle.load(f)
		if verbose:
			print_cfg(cfg)
		os.environ['CUDA_VISIBLE_DEVICE'] = cfg.cuda_dv
	return cfg

def pkl_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, 
						default = 'Pkl/test20.pkl', help = 'pkl file name')
	args = parser.parse_args()
	return args
	
if __name__ == '__main__':
	args = argparser()  # 解析命令行参数
	dump_pkl(args)  # 保存配置
	# cfg = load_pkl('./Pkl/test.pkl')
	# for k, v in vars(cfg).items():
	# 	print(k, v)
	# 	print(vars(cfg)[k])#==v
