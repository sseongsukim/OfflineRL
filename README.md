# OfflineRL

Offline reinforcement learning algorithms on [OGBench](https://github.com/seohongpark/ogbench).

`main.py` picks the setting from `--env_name` and asserts that the agent matches it:

- goal-conditioned agents: `crl`, `hiql`, `qrl`, `sharsa`, `trl`
- offline RL agents: `fql`, `rql`

## Usage

```bash
# Offline goal-conditioned RL.
python main.py --agent agents/hiql.py --env_name pointmaze-medium-navigate-v0

# Offline RL.
python main.py --agent agents/fql.py --env_name pointmaze-medium-navigate-singletask-task1-v0
```

Each agent declares the batch layout it needs through `dataset_class` in its `get_config()`; `main.py` looks the class
up in `utils/datasets.py`:

| `dataset_class` | Used by | What a batch contains |
| --- | --- | --- |
| `Dataset` | `fql` | single transitions |
| `SequenceDataset` | `rql` | length-`h + 1` sub-trajectories, for action chunking |
| `GCDataset` | `crl`, `qrl` | transitions plus `value_goals`/`actor_goals` |
| `HGCDataset` | `hiql` | `GCDataset` plus low-level subgoals and high-level subgoal targets |
| `SHARSADataset` | `sharsa` | `GCDataset` plus `subgoal_steps`-step high-level transitions |
| `TRLDataset` | `trl` | `GCDataset` plus a midpoint state between the state and the value goal |

Environments whose name contains `oraclerep` replace goals with oracle goal representations. `crl`, `trl`, and
`sharsa` read them from the dataset; `hiql` and `qrl` tie goals to raw observations and reject those environments.


## [RQL — Reversal Q-Learning](https://arxiv.org/pdf/2606.17551) (arXiv, 2026)

Reverses flows to construct virtual on-policy trajectories and trains a flow policy using multi-step value learning and behavior-regularized value maximization.

### Training Loss

**Value learning:**

$$
\mathcal{L}(V)=\mathbb{E}_{\tilde{\tau}}\left[
\ell_2^\kappa\left(V(s,x^f,f)-\left(r+\gamma V(s',x'^{0},0)\right)\right)
\right],
$$

where $\ell_2^\kappa(x)=|\kappa-\mathbb{I}(x\gt0)|x^2$.

**Policy learning:**

$$
\mathcal{L}(v)=
\underbrace{-\mathbb{E}_{\tilde{\tau}}\left[V(s,x^f+v(s,x^f,f),f+1)\right]}_{\text{value maximization}}
+\underbrace{\alpha\mathcal{L}^{\mathrm{BC}}(v)}_{\text{behavioral regularization}}.
$$

**Flow-matching regularizer:**

$$
\mathcal{L}^{\mathrm{BC}}(v)=
\mathbb{E}_{\substack{(s,a)\sim\mathcal{D},\\
x^0\sim\mathcal{N}(0,I_d),\\
f\sim\mathcal{U}(0,F)}}
\left[\left\lVert v(s,x^f,f)-\frac{1}{F}(a-x^0)\right\rVert_2^2\right],
$$

$$
x^f=(1-f/F)x^0+(f/F)a.
$$


## [TRL — Transitive RL: Value Learning via Divide and Conquer](https://arxiv.org/pdf/2510.22512) (ICLR, 2026)

Learns goal-conditioned values by composing shorter trajectory segments through behavioral subgoals, using expectile regression and distance-based re-weighting.

### Training Loss

**Value learning:**

$$
\mathcal{L}^{\mathrm{TRL}}(Q)=\mathbb{E}_{\tau\sim\mathcal{D}}\left[
w(s_i,s_j)D_\kappa\left(Q(s_i,a_i,s_j),\bar{Q}(s_i,a_i,s_k)\bar{Q}(s_k,a_k,s_j)\right)
\right].
$$

Here $\bar{Q}$ is the target Q function, and $s_k$ is a subgoal from the same trajectory,
with $0\leq i\lt j\lt T$ and $k\sim\mathrm{Unif}(\lbrace i,i+1,\ldots,j-1\rbrace)$.

**Distance-based re-weighting and expectile loss:**

$$
w(s_i,s_j):=\frac{1}{(1+\log_\gamma Q(s_i,a_i,s_j))^\lambda},
\qquad
D_\kappa(x,y):=|\kappa-\mathbb{I}(x\gt y)|D(x,y).
$$

The paper uses binary cross-entropy for $D$. For the base cases, replace $\bar{Q}(s_i,a_i,s_k)$
with $\gamma^{k-i}$ when $k-i\leq1$, and $\bar{Q}(s_k,a_k,s_j)$ with $\gamma^{j-k}$ when $j-k\leq1$.

**Policy extraction — reparameterized gradients:**

The paper's default policy extraction maximizes

$$
J^{\mathrm{DDPG+BC}}(\pi)=
\mathbb{E}_{\substack{s,a,g\sim\mathcal{D},\\
a^\pi\sim\pi(a\mid s,g)}}
\left[Q(s,a^\pi,g)+\alpha\log\pi(a\mid s,g)\right],
$$

where $a^\pi$ is a reparameterized policy sample and $\alpha$ controls behavioral regularization.

**Policy extraction — rejection sampling:**

$$
\pi(s,g)\overset{d}{=}
\underset{a_1,\ldots,a_N:\thinspace a_i\sim\pi^\beta(a\mid s,g)}{\arg\max}
Q(s,a_i,g),
$$

where $\pi^\beta$ is a goal-conditioned BC policy, $N$ is the number of samples, and
$\overset{d}{=}$ denotes equality in distribution.


## [SHARSA — Horizon Reduction Makes RL Scalable](https://arxiv.org/pdf/2506.04168) (NeurIPS, 2025)

Reduces both value and policy horizons by combining high-level n-step SARSA with hierarchical flow behavioral cloning and value-based subgoal selection.

### Training Loss

**High-level value learning:**

$$
L^V(\theta_V)=
\mathbb{E}_{\substack{(s_h,a_h,\ldots,s_{h+n})\sim p^{\mathcal{D}},\\
g\sim p^{\mathcal{D}}(g\mid s_h,a_h)}}
\left[D\left(V^h_{\theta_V}(s_h,g),Q^h_{\bar{\theta}_Q}(s_h,s_{h+n},g)\right)\right],
$$

$$
L^Q(\theta_Q)=
\mathbb{E}_{\substack{(s_h,a_h,\ldots,s_{h+n})\sim p^{\mathcal{D}},\\
g\sim p^{\mathcal{D}}(g\mid s_h,a_h)}}
\left[D\left(Q^h_{\theta_Q}(s_h,s_{h+n},g),
\sum_{i=0}^{n-1}\gamma^i r(s_{h+i},g)+\gamma^n V^h_{\theta_V}(s_{h+n},g)\right)\right].
$$

Here $n$ is the subgoal interval, $\bar{\theta}_Q$ denotes target-network parameters, and $D$ is either
$\mathrm{Reg}(x,y)=(x-y)^2$ or $\mathrm{BCE}(x,y)=-y\log x-(1-y)\log(1-x)$;
the paper uses BCE for its main experiments.

**High-level flow behavioral cloning:**

$$
L^h(\theta_h)=
\mathbb{E}_{\substack{(s_h,a_h,\ldots,s_{h+n})\sim p^{\mathcal{D}},\ g\sim p^{\mathcal{D}}(g\mid s_h,a_h),\\
z\sim\mathcal{N}(0,I_m),\ t\sim\mathrm{Unif}([0,1]),\\
w^t=(1-t)z+t\varphi_g(s_{t+h})}}
\left[\left\lVert v^h_{\theta_h}(t,s_h,w^t,g)-(\varphi_g(s_{t+h})-z)\right\rVert_2^2\right],
$$

where $\varphi_g$ is the goal specification function and $\mathcal{G}=\mathbb{R}^m$.
The $s_{t+h}$ index is retained exactly as printed in the paper's Appendix E.3.

**Low-level flow behavioral cloning:**

$$
L^\ell(\theta_\ell)=
\mathbb{E}_{\substack{(s_h,a_h,\ldots,s_{h+n})\sim p^{\mathcal{D}},\ z\sim\mathcal{N}(0,I_d),\\
t\sim\mathrm{Unif}([0,1]),\ a^t=(1-t)z+ta_h}}
\left[\left\lVert v^\ell_{\theta_\ell}(t,s_h,a^t,s_{h+n})-(a_h-z)\right\rVert_2^2\right],
$$

where $\mathcal{A}=\mathbb{R}^d$.

**Policy extraction — high-level rejection sampling:**

$$
\pi^h_{\theta_h}(s,g)\overset{d}{=}
\underset{w_1,\ldots,w_N:\thinspace w_i\sim\pi^h_{\beta,\theta_h}(w\mid s,g)}{\arg\max}
Q^h_{\theta_Q}(s,w_i,g),
$$

$$
\pi^\ell_{\theta_\ell}(s,w)\overset{d}{=}\pi^\ell_{\beta,\theta_\ell}(s,w).
$$

$N$ is the number of candidate subgoals; SHARSA directly uses the low-level BC policy.


## [FQL — Flow Q-Learning](https://arxiv.org/pdf/2502.02538) (ICML, 2025)

Trains a flow policy with behavioral cloning and a separate one-step policy that maximizes Q-values while staying close to the flow policy through distillation.

### Training Loss

**Critic learning:**

$$
\mathcal{L}_Q(\phi)=
\mathbb{E}_{\substack{s,a,r,s'\sim\mathcal{D},\\
a'\sim\pi_\omega}}
\left[(Q_\phi(s,a)-r-\gamma Q_{\bar{\phi}}(s',a'))^2\right],
$$

where $Q_{\bar{\phi}}$ is the target critic and $a'$ is sampled from the one-step policy at $s'$.

**Flow behavioral cloning:**

$$
\mathcal{L}_{\mathrm{Flow}}(\theta)=
\mathbb{E}_{\substack{s,a=x^1\sim\mathcal{D},\\
x^0\sim\mathcal{N}(0,I_d),\\
t\sim\mathrm{Unif}([0,1])}}
\left[\left\lVert v_\theta(t,s,x^t)-(x^1-x^0)\right\rVert_2^2\right],
$$

where $x^t=(1-t)x^0+tx^1$. The flow policy $\mu_\theta(s,z)=\psi_\theta(1,s,z)$ is the ODE output
induced by $v_\theta$, with $z\sim\mathcal{N}(0,I_d)$.

**One-step distillation:**

$$
\mathcal{L}_{\mathrm{Distill}}(\omega)=
\mathbb{E}_{\substack{s\sim\mathcal{D},\\
z\sim\mathcal{N}(0,I_d)}}
\left[\left\lVert\mu_\omega(s,z)-\mu_\theta(s,z)\right\rVert_2^2\right].
$$

**One-step policy learning:**

$$
\mathcal{L}_\pi(\omega)=
\underbrace{\mathbb{E}_{s\sim\mathcal{D},a^\pi\sim\pi_\omega}\left[-Q_\phi(s,a^\pi)\right]}_{\text{Q loss}}
+\underbrace{\alpha\mathcal{L}_{\mathrm{Distill}}(\omega)}_{\text{``BC'' loss}}.
$$

$\pi_\omega$ is the stochastic policy induced by $\mu_\omega(s,z)$, and $\alpha$ controls the distillation
regularizer. The flow policy learns only from BC; the one-step policy receives the Q gradient and is
used at test time.


## [HIQL — Offline Goal-Conditioned RL with Latent States as Actions](https://arxiv.org/pdf/2307.11949) (NeurIPS, 2023)

Learns an action-free goal-conditioned value function and extracts hierarchical policies with advantage-weighted regression, using learned subgoal representations to connect the two levels.

### Training Loss

**Action-free value learning:**

$$
\mathcal{L}_V(\theta_V)=
\mathbb{E}_{(s,s')\sim\mathcal{D}_S,\thinspace g\sim p(g\mid\tau)}
\left[L_2^\tau\left(r(s,g)+\gamma V_{\bar{\theta}_V}(s',g)-V_{\theta_V}(s,g)\right)\right],
$$

where $L_2^\tau(x)=|\tau-\mathbf{1}(x\lt0)|x^2$, $\tau\in[0.5,1)$ is the expectile parameter,
and $\bar{\theta}_V$ denotes target-network parameters. $\mathcal{D}_S$ contains state-only trajectories;
without additional action-free data, $\mathcal{D}_S=\mathcal{D}$.

**High-level policy learning:**

The policy objectives are maximized:

$$
J_{\pi^h}(\theta_h)=
\mathbb{E}_{(s_t,s_{t+k},g)}
\left[\exp\left(\beta\cdot\tilde{A}^h(s_t,s_{t+k},g)\right)
\log\pi^h_{\theta_h}(s_{t+k}\mid s_t,g)\right],
$$

$$
\tilde{A}^h(s_t,s_{t+k},g)=V_{\theta_V}(s_{t+k},g)-V_{\theta_V}(s_t,g).
$$

**Low-level policy learning:**

$$
J_{\pi^\ell}(\theta_\ell)=
\mathbb{E}_{(s_t,a_t,s_{t+1},s_{t+k})}
\left[\exp\left(\beta\cdot\tilde{A}^\ell(s_t,a_t,s_{t+k})\right)
\log\pi^\ell_{\theta_\ell}(a_t\mid s_t,s_{t+k})\right],
$$

$$
\tilde{A}^\ell(s_t,a_t,s_{t+k})=V_{\theta_V}(s_{t+1},s_{t+k})-V_{\theta_V}(s_t,s_{t+k}).
$$

Here $k$ is the subgoal step size and $\beta$ is the inverse temperature. Only low-level policy learning
requires action labels.

**Latent subgoals:**

The paper parameterizes the value function as $V(s,\phi(g))$ and uses $z_{t+k}=\phi(s_{t+k})$:
the high-level policy predicts $\pi^h(z_{t+k}\mid s_t,g)$, and the low-level policy uses
$\pi^\ell(a\mid s_t,z_{t+k})$. In the experiments, the representation additionally conditions on the
current state through $\phi([g,s])$.


## [CRL — Contrastive Learning as Goal-Conditioned Reinforcement Learning](https://arxiv.org/pdf/2206.07568) (NeurIPS, 2022)

Learns a goal-conditioned critic by contrasting future states against random states, then trains a policy to maximize the critic with behavioral regularization in the offline setting.

### Training Loss

**Contrastive critic learning — NCE:**

The paper maximizes the following binary contrastive objective:

$$
\max_f\mathbb{E}_{\substack{(s,a)\sim p(s,a),\thinspace s_f^-\sim p(s_f),\\
s_f^+\sim p^{\pi(\cdot\mid\cdot)}(s_{t+}\mid s_t,a_t)}}
\left[\mathcal{L}(s,a,s_f^+,s_f^-)\right],
$$

$$
\mathcal{L}(s,a,s_f^+,s_f^-)\triangleq
\log\sigma\left(\underbrace{f(s,a,s_f^+)}_{\phi(s,a)^T\psi(s_f^+)}\right)
+\log\left(1-\sigma\left(\underbrace{f(s,a,s_f^-)}_{\phi(s,a)^T\psi(s_f^-)}\right)\right).
$$

$\sigma$ is the sigmoid, $s_f^+$ is sampled from the discounted future-state distribution,
and $s_f^-$ from its marginal. The critic is parameterized as
$f(s,a,s_g)=\phi(s,a)^T\psi(s_g)$.

**Goal-conditioned policy learning:**

$$
\max_{\pi(a\mid s,s_g)}
\mathbb{E}_{\pi(a\mid s,s_g)p(s)p(s_g)}
\left[f(s,a,s_f=s_g)\right].
$$

**Offline policy learning — behavioral regularization:**

$$
\max_{\pi(a\mid s,s_g)}
\mathbb{E}_{\pi(a\mid s,s_g)p(s,a_{\mathrm{orig}},s_g)}
\left[(1-\lambda)\cdot f(s,a,s_f=s_g)
+\lambda\cdot\log\pi(a_{\mathrm{orig}}\mid s,s_g)\right].
$$

$a_{\mathrm{orig}}$ is the dataset action, while $a$ is sampled from the policy. $\lambda$ controls
the BC term; $\lambda=1$ gives goal-conditioned BC. The paper's offline experiments use multiple
critics and take their minimum for the actor update.


## [QRL — Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning](https://arxiv.org/abs/2304.01203) (ICML, 2023)

Learns optimal goal-reaching costs by maximizing quasimetric distances under local transition-cost constraints, then extracts a policy through learned latent transitions.

### Training Loss

**Quasimetric value learning:**

$$
\begin{gathered}
\min_\theta\max_{\lambda\geq0}
-\mathbb{E}_{\substack{s\sim p_{\mathrm{state}}\\
g\sim p_{\mathrm{goal}}}}
\left[\phi\left(d_\theta^{\mathrm{IQE}}(s,g)\right)\right]\\
+\lambda\left(
\mathbb{E}_{(s,a,s',r)\sim p_{\mathrm{transition}}}
\left[\mathrm{relu}\left(d_\theta^{\mathrm{IQE}}(s,s')+r\right)^2\right]
-\epsilon^2\right).
\end{gathered}
$$

$d_\theta^{\mathrm{IQE}}$ is an Interval Quasimetric Embedding model, $-r$ is the transition cost,
$\lambda$ is the Lagrange multiplier, and $\epsilon$ controls constraint relaxation.
$\phi$ shapes the distance-maximization term for stable optimization.

**Latent transition learning:**

$$
d_{\theta=(\theta_1,\theta_2)}(s_0,s_1)
\triangleq d^{\mathcal{Z}}_{\theta_1}\left(f_{\theta_2}(s_0),f_{\theta_2}(s_1)\right).
$$

Following the paper, omit parameter subscripts below and write
$z\triangleq f(s)$, $z'\triangleq f(s')$, $\hat z'\triangleq T(z,a)$, and $z_g\triangleq f(g)$.

$$
\mathcal{L}_{\mathrm{transition}}(s,a,s';T,d_\theta)
\triangleq\frac12\left(d^{\mathcal{Z}}(\hat z',z')^2+d^{\mathcal{Z}}(z',\hat z')^2\right).
$$

This loss jointly trains the latent transition model $T$ and quasimetric model $d_\theta$.

**Policy learning:**

$$
d^{\mathcal{Z}}(T(z,a),z_g)-r
=d^{\mathcal{Z}}(\hat z',z_g)-r
\approx-Q^*(s,a;g).
$$

The paper omits the constant transition cost in its experiments and trains the policy with

$$
\min_\pi\mathbb{E}_{\substack{s\sim p_{\mathrm{state}}\\
g\sim p_{\mathrm{goal}}}}
\left[d^{\mathcal{Z}}\left(T(f(s),a),f(g)\right)\right],
$$

where $a$ is generated by the policy. Its offline experiments additionally use behavioral cloning
regularization for policy learning.

## Acknowledgments

This codebase is built on top of [horizon-reduction](https://github.com/seohongpark/horizon-reduction) reference implementations.