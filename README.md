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
    python run_model_free.py --algo_name iql --env_name hopper-medium-v2 --batch_size 1024 --expectile 0.8 --temperature 3.0 --wandb_offline False
    ```
2. CQL
    ```terminal
    python run_model_free.py --algo_name cql --env_name hopper-medium-v2 --batch_size 1024 --wandb_offline False
    ```
3. TD3-BC
    ```terminal
    python run_model_free.py --algo_name td3bc --env_name hopper-medium-v2 --batch_size 1024 --alpha 2.5 --noise_clip 0.5 --update_freq 2 --wandb_offline False
    ```
--- 
### References
1. [Jaxrl_m](https://github.com/dibyaghosh/jaxrl_m)
2. [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit)
3. [robot-learning](https://github.com/youngwoon/robot-learning)
4. [LEQ](https://github.com/kwanyoungpark/LEQ)
5. [HIQL](https://github.com/seohongpark/HIQL)
   
