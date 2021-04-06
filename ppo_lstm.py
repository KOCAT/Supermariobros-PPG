##ppg

import numpy as np
import torch
import torch.nn as nn
from torch import distributions as td
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from collections import deque, namedtuple
import gym
import time
import scipy.signal
from core_lstm import *
from env import *
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

#device = torch.device('cuda')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

B = namedtuple('B',['obs', 'ret', 'act', 'adv', 'logp_old','logp', 'hidden_h', 'hidden_c'])
to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))

def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, shuffle = True)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPGBuffer:


    def __init__(self, obs_dim, act_dim, hidden_dim, size, gamma=0.99, lam=0.95, beta_s=0.01):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)      
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.hidden_h_buf = np.zeros(combined_shape(size, hidden_dim), dtype=np.float32)
        self.hidden_c_buf = np.zeros(combined_shape(size, hidden_dim), dtype=np.float32)
        print(self.hidden_h_buf.shape)
        self.gamma, self.lam, self.beta_s = gamma, lam, beta_s
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, hidden_h, hidden_c):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.hidden_h_buf[self.ptr] = hidden_h
        self.hidden_c_buf[self.ptr] = hidden_c
        self.ptr += 1



    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, rew=self.rew_buf, val=self.val_buf,
                    adv=self.adv_buf, logp=self.logp_buf, hidden_h=self.hidden_h_buf, hidden_c=self.hidden_c_buf)
        #data.to(device)
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in data.items()}



def ppg(env_fn, actor=nn.Module, critic=nn.Module, ac_kwargs=dict(), seed=0, epochs_aux=6, beta_clone=1,
        steps_per_epoch=4000, epochs=50, gamma=0.999, beta_s=0.01, clip_ratio=0.2, minibatch_size=16,
        lr=5e-4, lr_aux=5e-4, train_pi_iters=1, train_v_iters=1, n_pi=32, lam=0.95, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=10, pretrain=None,):


    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n
    #act_dim = env.action_space.shape #原代码
    
    # Create actor-critic module
    if pretrain != None:
        ac_pi = torch.load(pretrain)
    else:
        ac_pi = actor(obs_dim[0], act_dim, hidden_sizes=[64, 64], activation=nn.Tanh, pretrain=pretrain)  # env.observation_space, env.action_space, nn.ReLU)
    ac_v = critic(obs_dim[0], hidden_sizes=[64, 64], activation=nn.Tanh)  # env.observation_space, nn.ReLU)

    #device = torch.device('cuda')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ac_pi.to(device)
    ac_v.to(device)
    #ac_pi = nn.DataParallel(ac_pi)
    #ac_v = nn.DataParallel(ac_v)

    # Sync params across processes
   # sync_params(ac_pi)
   # sync_params(ac_v)

    # Count variables
    def count_vars(module):
        return sum([np.prod(p.shape) for p in module.parameters()])
    var_counts = tuple(count_vars(module) for module in [ac_pi, ac_v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPGBuffer(obs_dim, env.action_space.shape, 512, local_steps_per_epoch, gamma, lam, beta_s)





    def compute_loss_pi(data):
        obs, act, adv, logp_old , hidden_h, hidden_c = data['obs'], data['act'], data['adv'], data['logp'], data['hidden_h'], data['hidden_c']

        pi, logp ,_ = ac_pi(obs, act, (hidden_h, hidden_c))
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)+beta_s * pi.entropy()).mean()
        return loss_pi

    def compute_loss_v(data):
        obs, ret ,hidden_h, hidden_c = data['obs'], data['ret'], data['hidden_h'], data['hidden_c']
        loss_v = (0.5*(ac_v(obs, (hidden_h, hidden_c)) - ret)**2).mean()
        return loss_v
     
    def compute_loss_aux(obs, ret ,hidden_h, hidden_c):
        loss_aux = (0.5*(ac_v(obs, (hidden_h, hidden_c)) - ret)**2).mean()
        return loss_aux

    def compute_loss_joint(obs, act, adv, logp_old , hidden_h, hidden_c):
        pi, logp ,_ = ac_pi(obs, act, (hidden_h, hidden_c))
        loss_aux = compute_loss_aux(obs, ret ,hidden_h, hidden_c)
        loss_kl = td.kl_divergence(logp_old, logp).mean()
        policy_loss = aux_loss + beta_clone*loss_kl
        return joint_loss   


    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac_pi.parameters(), lr=lr)
    vf_optimizer = Adam(ac_v.parameters(), lr=lr)
    joint_optimizer = Adam(ac_pi.parameters(), lr=lr_aux)
    aux_optimizer = Adam(ac_v.parameters(), lr=lr_aux)


    # Set up model saving
    logger.setup_pytorch_saver(ac_pi)
    

        
    def update():


        pi_l_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()                   

        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            mpi_avg_grads(ac_pi)    
            pi_optimizer.step()
            
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac_v)    
            vf_optimizer.step()
        
        '''
        obs = to_torch_tensor(obs)
        ret = to_torch_tensor(ret)
        act = to_torch_tensor(act)
        adv = to_torch_tensor(adv)
        logp_old = to_torch_tensor(logp_old)
        logp = to_torch_tensor(logp)
        hidden_h = to_torch_tensor(hidden_h)
        hidden_c = to_torch_tensor(hidden_c)
        '''
        '''

        '''
        
       
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
    
    def updateaux():
        for i in range(epochs_aux):
            obss = []
            rets = []
            acts = []
            advs = []
            logp_olds = []
            logps = []
            hidden_hs = []
            hidden_cs = []
            for obs, ret, act, adv, logp_old , hidden_h, hidden_c in aux_memories:
                obss.append(obs)
                rets.append(ret)
                acts.append(act)
                advs.append(adv)
                logp_olds.append(logp_old)
                logps.append(logp)
                hidden_hs.append(hidden_h)                
                hidden_cs.append(hidden_c)   
            
            obss = torch.cat(obss)
            rets = torch.cat(rets)
            acts = torch.cat(acts)
            advs = torch.cat(advs)
            logp_olds = torch.cat(logp_olds)
            logps = torch.cat(logps)
            hidden_hs = torch.cat(hidden_hs)
            hidden_cs = torch.cat(hidden_cs)            
            
            dl = create_shuffled_dataloader([obss,rets,acts,advs,logp_olds,logps,hidden_hs,hidden_cs],batch_size=minibatch_size)
            for obs, ret, act, adv, logp_old , hidden_h, hidden_c in dl:                             
                joint_optimizer.zero_grad()
                loss_joint = compute_loss_joint(obs, act, adv, logp_old , hidden_h, hidden_c)
                loss_joint.backward()
                mpi_avg_grads(ac_pi)    
                joint_optimizer.step()

                aux_optimizer.zero_grad()
                loss_aux = compute_loss_aux(obs, ret ,hidden_h, hidden_c)
                loss_aux.backward()
                mpi_avg_grads(ac_v)    
                aux_optimizer.step()


    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    aux_memories = deque([])
    
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        hidden = (torch.zeros((1, 512), dtype=torch.float).to(device), torch.zeros((1, 512), dtype=torch.float).to(device))
        for i in range(n_pi):
            for t in range(local_steps_per_epoch):
                # a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
                with torch.no_grad():
                    rr = torch.from_numpy(o.copy()).float().to(device)#.unsqueeze(0)
                    pi, _, hidden_ = ac_pi(rr, None, hidden)
                    a = pi.sample()
                    # logp_a = self.pi._log_prob_from_distribution(pi, a)
                    logp = pi.log_prob(a)#.sum(axis=-1)
                    v = ac_v(torch.as_tensor(o, dtype=torch.float32).to(device), hidden)

                next_o, r, d, _ = env.step(a.cpu().numpy().item())
                ep_ret += r
                ep_len += 1

                # save and log
                #print(hidden[0].shape)
                buf.store(o, a.cpu().numpy(), r, v.cpu().numpy(), logp.cpu().numpy(), hidden[0].cpu().numpy(), hidden[1].cpu().numpy())
                logger.store(VVals=v.cpu().numpy())
                
                # Update obs (critical!)
                o = next_o
                hidden = hidden_

                timeout = ep_len == max_ep_len
                terminal = d #or timeout
                epoch_ended = t==local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by n_pi at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        print('epoch_end')
                        # _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                        with torch.no_grad():
                            v =ac_v(torch.from_numpy(o).float().to(device), hidden).cpu().numpy()
                    else:
                        print('epret :',ep_ret)
                        v = 0
                        hidden= (torch.zeros((1, 512), dtype=torch.float).to(device), torch.zeros((1, 512), dtype=torch.float).to(device))
                    buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = env.reset(), 0, 0


            # Save model
            if (i % save_freq == 0) or (i == i):
                logger.save_state({'env': env}, None)
            
            
            data = buf.get()
            obs, ret, act, adv, logp_old , hidden_h, hidden_c = data['obs'], data['ret'], data['act'], data['adv'], data['logp'], data['hidden_h'], data['hidden_c']
            pi, logp ,_ = ac_pi(obs, act, (hidden_h, hidden_c))
            aux_memory = B(obs, ret, act, adv, logp_old, logp, hidden_h, hidden_c)
            aux_memories.append(aux_memory)
            # Perform PPG update!
            update()
        
        updateaux()    
                

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v2')#'HalfCheetah-v2')
    parser.add_argument('--world',type=str, default='1')
    parser.add_argument('--stage',type=str, default='1')
    parser.add_argument('--actiontype',type=str, default='complex')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=350)
    #parser.add_argument('--pretrain', type=str, default='pretrain/ppg_lstm/pyt_save/model.pt')
    parser.add_argument('--pretrain', type=str, default='spinningup/data/race/race_s0/pyt_save/model.pt')
    parser.add_argument('--exp_name', type=str, default='race')
    args = parser.parse_args()

#    import os
#    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    env_fn = lambda : create_train_env(args.world, args.stage, args.actiontype)
    ppg(env_fn, actor=userActor, critic=userCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs,
        minibatch_size=16, clip_ratio=0.2, lr=0.0005, lr_aux=0.0005,beta_clone=1,
        train_pi_iters=1, train_v_iters=1,epochs_aux=6, beta_s=0.01, n_pi=32 ,pretrain=None)#args.pretrain)

