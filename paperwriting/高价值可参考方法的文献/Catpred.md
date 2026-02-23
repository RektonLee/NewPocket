下面我就按你最关心的三件事来“拆开讲懂”这篇 **CatPred（Nat. Commun. 2025）**：**(A) 数据怎么做、(B) 模型怎么做、(C) 训练/推理的flow与张量维度**，并且结合他们的官方 GitHub 仓库（有的）。你上传的论文我也会逐段引用。

---

## 0) 这篇到底在解决什么问题（从AI视角）

CatPred 的目标不是“再做一个回归模型”，而是同时解决三个痛点：

1. **数据不标准**：BRENDA / SABIO-RK 里很多条目缺序列/底物映射/重复测量冲突；不同工作过滤规则不一致导致不可比。论文在 Methods 里强调要做完整注释、统一 SMILES 映射与去重策略。
2. **泛化评测要“真冷启动”**：不仅 test 不含同一 enzyme-substrate pair，还要把与训练集高度同源的序列剔除，构造 OOD set（例如 max seq id ≤99% 甚至降到 40%）。
3. **要有不确定性（UQ）**：输出不是一个点值，而是一个“预测分布”（mean + variance），并用 ensemble 给 epistemic。

---

## 1) 数据处理与数据结构（你做 pipeline 会最关心的部分）

### 1.1 样本定义（一个 data point 是什么）

一个样本基本是：

* **enzyme**：UniProt 对应的氨基酸序列（可选：对应的 3D 结构 embedding）
* **ligand side**：

  * 做 **kcat**：把 *该反应的所有 reactants* 的 SMILES **拼接成一个串**（concatenated SMILES），因为他们发现共底物/辅因子对 kcat 有信息；
  * 做 **Km / Ki**：只用“对应底物/抑制剂”的 SMILES（不需要反应物集合）
* **label**：log10 变换后的 kcat / Km / Ki（论文多处强调训练是在 log10 空间做回归，UQ 的 SD 也在 log10 空间）。

### 1.2 重复测量如何压成一个 label（这点非常关键）

同一个（sequence, SMILES）可能有多条测量：

* **kcat**：取 **最大值**（他们认为最大值更接近最优条件，如温度/pH）
* **Km / Ki**：取 **几何平均**（等价于对 log 值做算术平均）

> AI角度理解：这是在减少“同输入多输出”的标签冲突，否则回归会在训练中震荡，尤其是高维表征。

### 1.3 SMILES 到图：原子/键特征（你要对齐实现）

论文在 Methods 的 “Deep learning architecture” 段明确写了 atom / bond 的 one-hot 特征构成：

* atom：atomic number、#bonds、formal charge、hybridization、aromaticity、atomic mass、#H、chirality（全部 one-hot 后 concat）
* bond：bond type（single/double/triple/aromatic）、conjugation、ring、bond chirality（one-hot concat）
  然后用 **D-MPNN** 把这些转成 molecule embedding。

### 1.4 数据集划分（pair-level + sequence OOD）

* 常规 held-out：80/10/10 split，但**保证同一 enzyme-substrate pair 不跨分区**（kcat 是 enzyme + reactant-set）。
* OOD：从 held-out test 里再抽子集，并保证 test 序列与训练集序列最大同一性 ≤99%（后面还做 80/60/40% 分档）。序列聚类工具是 **mmseqs2**。

---

## 2) 模型结构：三路 enzyme 表征 + 一路 substrate 表征 + 概率回归头

CatPred 的设计思路是“你可以逐步加信息，看哪一部分真的帮泛化”：

* enzyme 侧：Seq-Attn（自己学） + pLM（ESM-2 抽特征） + （可选）结构 E-GNN embedding
* ligand 侧：D-MPNN
  最后 concat 过 MLP 输出 mean/variance。

下面我按实现逻辑讲每一块。

---

## 3) 关键：数据 flow 与张量维度（按 forward 走一遍）

我用符号：

* batch size = **B**
* enzyme 序列长度（padding 后）= **L**
* 反应/底物分子图：每个分子 i 有原子数 **Nᵢ**、键数 **Eᵢ**

### 3.1 enzyme：Seq-Attn 分支（可训练）

论文图 2a 给了形状暗示：输入是 `[L × 1]` 的 token 序列，经过 embedding 和 attention 后变成 `[L × 36]`。

典型实现会是：

1. tokenization：把氨基酸映射到整数 id

   * `seq_ids`: **[B, L]**（pad 到 L）
2. embedding：`nn.Embedding(vocab, d_emb)`

   * `x0`: **[B, L, d_emb]**
3. Rotary positional embedding（RoPE）：对 Q/K 注入位置，论文明确用了 RoPE。
4. Multi-head self-attention 堆叠 N 层：

   * `x_att`: **[B, L, d_att]**（论文示意 d_att≈36）

> 你在代码里通常会看到：`d_att = n_heads * head_dim`；他们具体超参在 Supplementary Table 4（主文只说会调 embedding dim、RoPE dim、attention layers、attentive pooling layers）。

### 3.2 enzyme：pLM（ESM-2）分支（固定特征 or 半固定）

他们用 **ESM-2 650M**：`esm2_t33_650M_UR50D`，并提到每条序列抽出来是 **1280-dim** 特征。

常见做法有两种（仓库里通常会写清楚）：

* **方式 A：per-residue embedding**：输出 **[B, L, 1280]**
* **方式 B：sequence embedding**：对 residue embedding 做 mean/cls pooling 得到 **[B, 1280]**

CatPred 主文描述是“抽取每个 enzyme sequence 的 1280-dim features”，并且还提到后续把这些与 Seq-Attn/embedding 拼接后做 attentive pooling。
所以更像是 **per-residue 特征拼到 token 维度上**：

* `x_plm`: **[B, L, 1280]**
* `x_concat_token = concat([x_emb, x_att, x_plm], dim=-1)`：**[B, L, d_total]**

### 3.3 enzyme：Attentive pooling（把变长序列压成定长向量）

论文明确写：用一个 attentive pooling，学习每个 position 的权重，然后对长度维做加权平均，得到“final enzyme representations”。

实现上通常是：

* `α = softmax(W·tanh(V x_concat_token))`：**[B, L, 1]**
* `e = Σ α ⊙ x_concat_token`：**[B, d_total]**

这一步很关键：它把变长序列统一到一个向量，后面才能跟 molecule embedding concat。

### 3.4 enzyme：结构 E-GNN 分支（可选，用于结构特征）

他们直接用 Greener 等人的预训练 E-GNN（不改权重）抽结构 embedding，每个结构 **128-dim**。

所以结构向量是：

* `e_struct`: **[B, 128]**

并且论文结论是：

* **kcat/Km**：加 E-GNN 不明显（pLM 已含结构信息），最终 production 用 **Substrate + Seq-Attn + pLM**；
* **Ki**：pLM 1280 维容易在小数据集上过拟合，改用 **E-GNN 128 维**反而更稳，最终 production 用 **Substrate + Seq-Attn + E-GNN**。

### 3.5 ligand：D-MPNN 分支（核心是“有向边消息传递”）

他们按 D-MPNN 的标准流程：

1. RDKit 从 SMILES 建图（原子/键 one-hot）
2. 构造 **directed edge feature**：把“起点原子特征 + 键特征”拼起来作为一条有向边的初始表征；
3. T 次 message passing：对每条有向边聚合邻居边信息并更新
4. 读出：论文写“最终分子表示由所有 atom features 求和得到”（sum over atoms）。

论文图 2b 把最终 substrate embedding 画成 `[1 × 300]`，你可以把它理解为 `m ∈ R^{B×300}`。

### 3.6 拼接 + 概率回归头（输出 mean & variance）

最后把 enzyme 向量和 ligand 向量拼起来：

* `z = concat([e, e_struct(optional), m], dim=-1)`：**[B, d_z]**
  然后过一个 MLP，输出两个实数：
* `μ`: **[B, 1]**
* `logσ²` 或 `σ²`: **[B, 1]**（实现通常会输出 log-variance 再 softplus，保证方差正）

论文明确说：输出 “two real values representing the mean and the variance”。

---

## 4) 训练：NLL + ensemble（怎么把 UQ 做出来）

### 4.1 单模型：高斯负对数似然（NLL）

他们不是用 MSE，而是用高斯 NLL：每个样本的预测是 `Normal(μ, σ²)`，训练最小化 NLL。

（实现提示）如果网络直接输出 `σ²`，要用 `σ² = softplus(s) + eps` 这种方式避免数值炸掉。

### 4.2 Aleatoric vs Epistemic 怎么算

论文给了很清晰的公式描述：

* 每个模型给 `(μ_i, σ_i²)`；其中 `σ_i²` 对应 **aleatoric**（数据噪声）
* 训练 **10 个不同随机初始化**的同构网络作为 ensemble（N=10）。
* 最终预测：

  * `μ_final = mean_i μ_i`
  * `σ_epi = std_i μ_i`
  * `σ_ale = sqrt(mean_i σ_i²)`
  * `σ_total = σ_ale + σ_epi`（论文图 6 的流程是“sum of contributions”，注意都在 log10 空间）。

> 这套做法的直观解释：
>
> * **aleatoric**：模型认为“这个点的数据本身就很噪”，所以单模型也给大方差；
> * **epistemic**：不同模型对这个点意见不合（因为训练数据覆盖不足），所以均值分散大。

### 4.3 超参数与训练设置（主文给到的）

主文 Methods 里能直接抠出来的训练要点：

* substrate(D-MPNN) 的超参采用参考实现的推荐最优值；
* 学习率 `1e-3`；batch size 会调；Adam；ReLU；训练 20 epochs；production 模型每个 replicate 训练 30 epochs；ensemble size=10。

---

## 5) 官方 GitHub 仓库：你可以从哪里直接看实现（建议你优先看的文件）

他们公开了两个关键仓库：

1. **CatPred（模型与推理代码）**：`maranasgroup/CatPred` ([GitHub][1])
2. **CatPred-DB（数据与脚本）**：`maranasgroup/CatPred-DB` ([GitHub][2])

如果你想最快把“输入→输出”跑通，建议从 `demo_run.py` 入手（它就是“加载预训练模型→处理输入→输出预测与不确定性”的脚本）。([GitHub][3])
另外他们也把可复现实验环境放在 Code Ocean。([codeocean.com][4])

---

## 6) 你如果要“复刻成你自己的工程代码”，我建议你按这份 checklist 对齐

（按最容易踩坑排序）

1. **kcat 的 SMILES 一定是 reactants concat**；Km/Ki 只用单底物/抑制剂 SMILES。
2. **label 一定是 log10 空间训练**，所以你预测出来的 μ、σ 都在 log10 空间（回到原始单位要 `10^μ`；区间/SD 要相应变换）。
3. **attentive pooling** 是把 token-level 特征压成 sequence-level 的关键，否则你很难稳定地拼接 molecule embedding。
4. **Ki 的特征选择**：小数据集别硬上 1280-dim pLM，CatPred 自己就观察到会过拟合，最终 Ki production 换成 E-GNN 128-dim。

---

如果你下一步想更“落地到代码层面”，你把 **GitHub 仓库里你最关心的文件路径**（比如 `demo_run.py`、`models/*.py`、`data/*.py`）贴出来（或直接把仓库打包上传），我可以按文件逐段带你走：

* 输入文件格式长什么样
* 每一步生成了哪些张量（shape/dtype）
* D-MPNN 的 batch 组织（图的打包方式）
* ESM2 是离线预计算还是在线 forward（对速度影响巨大）

（你也可以告诉我：你更关心“训练复现”还是“推理部署/批量预测”，我会用不同的讲法带你看代码。）

[1]: https://github.com/maranasgroup/CatPred?utm_source=chatgpt.com "maranasgroup/CatPred: Machine Learning models for in ..."
[2]: https://github.com/maranasgroup/CatPred-DB?utm_source=chatgpt.com "maranasgroup/CatPred-DB"
[3]: https://github.com/maranasgroup/CatPred/blob/main/demo_run.py?utm_source=chatgpt.com "demo_run.py - maranasgroup/CatPred"
[4]: https://codeocean.com/capsule/9821966/tree/v1?utm_source=chatgpt.com "CatPred: A comprehensive framework for deep learning in ..."
