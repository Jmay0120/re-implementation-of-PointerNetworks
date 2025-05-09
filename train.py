import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 提供进度条显示
from time import time
from datetime import datetime  # 记录时间戳

from actor import PtrNet1  # 强化学习机制，Actor基于当前策略，生成一个城市访问序列
from critic import PtrNet2  # Critic预测该路径的代价（长度），与真实长度进行比较，计算损失，进而帮助Actor学习更优的路径策略
from env import Env_tsp
from config import Config, load_pkl, pkl_parser
from data import Generator

# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

def train_model(cfg, env, log_path = None):
	date = datetime.now().strftime('%m%d_%H_%M')
	if cfg.islogger:
		param_path = cfg.log_dir + '%s_%s_param.csv'%(date, cfg.task)# cfg.log_dir = ./Csv/
		print(f'generate {param_path}')
		with open(param_path, 'w') as f:
			f.write(''.join('%s,%s\n'%item for item in vars(cfg).items()))

	act_model = PtrNet1(cfg)  # 初始化Actor模型
	if cfg.optim == 'Adam':  # 使用Adam优化器，并根据需求应用学习率衰减
		act_optim = optim.Adam(act_model.parameters(), lr = cfg.lr)
	if cfg.is_lr_decay:
		act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, 
						step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	act_model = act_model.to(device)

	if cfg.mode == 'train':
		cri_model = PtrNet2(cfg)
		if cfg.optim == 'Adam':
			cri_optim = optim.Adam(cri_model.parameters(), lr = cfg.lr)
		if cfg.is_lr_decay:
			cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, 
						step_size = cfg.lr_decay_step, gamma = cfg.lr_decay)
		cri_model = cri_model.to(device)
		ave_cri_loss = 0.

	mse_loss = nn.MSELoss()
	dataset = Generator(cfg, env)  # 生成了n_samples个样本，每个样本里有20个城市，每个城市一个坐标点
	dataloader = DataLoader(dataset, batch_size = cfg.batch, shuffle = True)

	ave_act_loss, ave_L = 0., 0.
	min_L, cnt = 1e7, 0
	t1 = time()
	# for i, inputs in tqdm(enumerate(dataloader)):
	for i, inputs in enumerate(dataloader):
		inputs = inputs.to(device)
		pred_tour, ll = act_model(inputs, device)  # 预测路径
		real_l = env.stack_l_fast(inputs, pred_tour)  # 实际路径长度
		if cfg.mode == 'train':
			pred_l = cri_model(inputs, device)  # Critic预测的长度
			cri_loss = mse_loss(pred_l, real_l.detach())  # MSE损失
			cri_optim.zero_grad()
			cri_loss.backward()
			nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm = 1., norm_type = 2)
			cri_optim.step()
			if cfg.is_lr_decay:
				cri_lr_scheduler.step()
		elif cfg.mode == 'train_emv':
			if i == 0:
				L = real_l.detach().mean()
			else:
				L = (L * 0.9) + (0.1 * real_l.detach().mean())
			pred_l = L

		adv = real_l.detach() - pred_l.detach()
		act_loss = (adv * ll).mean()  # Actor损失
		act_optim.zero_grad()
		act_loss.backward()
		nn.utils.clip_grad_norm_(act_model.parameters(), max_norm = 1., norm_type = 2)
		act_optim.step()
		if cfg.is_lr_decay:
			act_lr_scheduler.step()

		ave_act_loss += act_loss.item()
		if cfg.mode == 'train':
			ave_cri_loss += cri_loss.item()
		ave_L += real_l.mean().item()
		
		if i % cfg.log_step == 0:
			t2 = time()
			if cfg.mode == 'train':	
				print('step:%d/%d, actic loss:%1.3f, critic loss:%1.3f, L:%1.3f, %dmin%dsec'%(i, cfg.steps, ave_act_loss/(i+1), ave_cri_loss/(i+1), ave_L/(i+1), (t2-t1)//60, (t2-t1)%60))
				if cfg.islogger:
					if log_path is None:
						log_path = cfg.log_dir + '%s_%s_train.csv'%(date, cfg.task)#cfg.log_dir = ./Csv/
						with open(log_path, 'w') as f:
							f.write('step,actic loss,critic loss,average distance,time\n')
					else:
						with open(log_path, 'a') as f:
							f.write('%d,%1.4f,%1.4f,%1.4f,%dmin%dsec\n'%(i, ave_act_loss/(i+1), ave_cri_loss/(i+1), ave_L/(i+1), (t2-t1)//60, (t2-t1)%60))
			
			elif cfg.mode == 'train_emv':
				print('step:%d/%d, actic loss:%1.3f, L:%1.3f, %dmin%dsec'%(i, cfg.steps, ave_act_loss/(i+1), ave_L/(i+1), (t2-t1)//60, (t2-t1)%60))
				if cfg.islogger:
					if log_path is None:
						log_path = cfg.log_dir + '%s_%s_train_emv.csv'%(date, cfg.task)#cfg.log_dir = ./Csv/
						with open(log_path, 'w') as f:
							f.write('step,actic loss,average distance,time\n')
					else:
						with open(log_path, 'a') as f:
							f.write('%d,%1.4f,%1.4f,%dmin%dsec\n'%(i, ave_act_loss/(i+1), ave_L/(i+1), (t2-t1)//60, (t2-t1)%60))
			if(ave_L/(i+1) < min_L):
				min_L = ave_L/(i+1)
				
			else:
				cnt += 1
				print(f'cnt: {cnt}/20')
				if(cnt >= 20):  # 如果连续20次损失没有改善，则停止训练
					print('early stop, average cost cant decrease anymore')
					if log_path is not None:
						with open(log_path, 'a') as f:
							f.write('\nearly stop')
					break
			t1 = time()
	if cfg.issaver:  # 训练结束后，将模型参数保存为.pt文件
		torch.save(act_model.state_dict(), cfg.model_dir + '%s_%s_step%d_act.pt'%(cfg.task, date, i))#'cfg.model_dir = ./Pt/'
		print('save model...')

if __name__ == '__main__':
	cfg = load_pkl(pkl_parser().path)  # 加载配置文件
	env = Env_tsp(cfg)  # 初始化TSP环境

	if cfg.mode in ['train', 'train_emv']:
		# train_emv --> exponential moving average, not use critic model
		train_model(cfg, env)  # 开始训练
	else:
		raise NotImplementedError('train and train_emv only, specify train pkl file')
				
