# OfflineRL
## Implement offline reinforcement learning algorithms (both Model-free and Model-based)
### Model-free Algorithms
- [CQL](https://arxiv.org/abs/2006.04779)
- [IQL](https://arxiv.org/abs/2110.06169)
- [TD3-BC](https://arxiv.org/pdf/2106.06860)
---
### Model-based Algorithms
- [MOPO](https://arxiv.org/abs/2005.13239)
- [RAMBO-RL](https://arxiv.org/abs/2204.12581)
- [MOBILE](https://proceedings.mlr.press/v202/sun23q.html)
---
### Software Environment
1. OS: Ubuntu 20.04 (Available in 22.04 as well)
2. CUDA: 12.6 / CUDNN: 8.9.6
--- 
### Installation
1. Install MuJoCo and then install mujoco-py. I recommend following [this](https://docs.google.com/document/u/1/d/1eBvfKoczKmImUgoGMbqypODBXmI1bD91/edit) document.
2. Install D4RL:
    ```terminal
    pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
    ```
3. Install our requirement.txt
    ```terminal
    pip install -r requirement.txt
    ```
---
### Train Model-free algorithms
1. IQL
```terminal
python run_model_free.py --algo_name iql --env_name hopper-medium-v2 --batch_size 1024 --expectile 0.8 --temperature 3.0 --wandb_offline 0
```
2. CQL
```terminal
python run_model_free.py --algo_name cql --env_name hopper-medium-v2 --batch_size 1024 --wandb_offline 0
```
3. TD3-BC
```terminal
python run_model_free.py --algo_name td3bc --env_name hopper-medium-v2 --batch_size 1024 --alpha 2.5 --noise_clip 0.5 --update_freq 2 --wandb_offline 0
```
--- 
### Train Dynamics
To train model-based algorithms, we first train a dynamics model that predicts the next state and reward based on the current observations and actions $\hat{T}(s',r|s,a)$.
``` terminal
python run_dynamics.py --env_name hopper-medium-v2 --batch_size 1024 --wandb_offline 0
```
---
### Train model-based algorithms
1. MOPO
```terminal
python run_model_based.py --env_name hopper-medium-v2 --algo_name mopo --batch_size 1024 --wandb_offline 0 --penalty_coef 0.5 --dataset_ratio 0.15 --dynamics_params_path {Your dynamics path}
```
2. RAMBO
```terminal
python rum_model_based.py --env_name hopper-medium-v2 --algo_name rambo --batch_size 1024 --wandb_offline 0 --penalty_coef 0.5 --dataset_ratio 0.5 --adv_weights 0.0003 --dynamics_update_freq 1000 --dynamics_params_path {Your dynamics path}
```
3. MOBILE
```terminal
python run_model_based.py --env_name hopper-medium-v2 --algo_name mobile --batch_size 1024 --wandb_offline 0 --penalty_coef 1.0 --dynamics_params_path {Your dynamics path}
```
### References
1. [Jaxrl_m](https://github.com/dibyaghosh/jaxrl_m)
2. [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit)
3. [LEQ](https://github.com/kwanyoungpark/LEQ)
4. [HIQL](https://github.com/seohongpark/HIQL)
   
