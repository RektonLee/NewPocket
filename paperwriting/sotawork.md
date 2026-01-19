下面按“**近期（2023–2025/early 2026）酶动力学参数 kcat / Km 预测**”这一细分方向，把真正经常被当作 **SOTA / 强基线** 的工作梳理成一条清晰主线：**数据 → 表征 → 结构信息 → 泛化评测 → 不确定性/可用性**。我会优先放**同行评审**论文，其次放影响力较大的 **bioRxiv/预印本**。

---

## 1) 目前最像“社区基准 + 强SOTA框架”的：CatPred（Nat. Commun. 2025）

**CatPred: a comprehensive framework for deep learning in vitro enzyme kinetic parameters**（Nature Communications, 2025-02-28）([Nature][1])

为什么它像“SOTA框架”：

* **任务覆盖**：同时做 *kcat / Km / Ki*，而不是只做 kcat。([Nature][1])
* **数据贡献很硬**：提出/整理了较大覆盖的 benchmark 数据集（文中给出量级：kcat ~23k、Km ~41k、Ki ~12k）。([Nature][1])
* **重点抓泛化**：强调在**训练集不相似序列**上的表现、以及**不确定性估计**（预测方差与误差相关）。([Nature][1])
* **模态更全**：输入可以用序列/结构 + 底物 SMILES（对 kcat 会把所有反应物 SMILES 拼接；对 Km/Ki 用对应底物的 SMILES）。([Nature][1])

你如果想做论文里的“对标 SOTA”，CatPred 很适合作为**核心对照**：它不仅是模型，还提供了更标准的数据与评测叙事。

---

## 2) “严肃冷启动评测（低同源）+ 序列/底物PLM组合”的：CataPro（Nat. Commun. 2025）

**Robust enzyme discovery and engineering with deep learning using CataPro**（Nature Communications, 2025-03-20）([Nature][2])

它的亮点在于**评测设计**更“工程化”：

* 用 **CD-HIT** 按序列相似度（文中提到 cutoff 0.4）把数据分组做更“unbiased”的十折评测，强调模型对低同源的泛化。([Nature][2])
* 表征上是典型的 **蛋白 PLM embedding（ProtT5）+ 底物 embedding（MolT5）+ 指纹（MACCS）** 拼接，再用 MLP/NN 做回归。([Nature][2])
* 还做了 **kcat/Km**：先分别预测 kcat 和 Km，再用“修正项”网络纠偏（避免把 kcat/Km 当作完全独立任务）。([Nature][2])

如果你关心“**真实应用里最怕的：新酶/低同源**”，CataPro 的分组/评测策略值得直接借鉴成你论文的实验设置。

---

## 3) “多任务 + 把环境因素纳入输入”的强工作：MPEK（2024, OUP/PubMed）

**MPEK: a multitask deep learning framework …**（2024）([PubMed][3])

它解决了一个被很多工作忽略但非常关键的问题：**kcat/Km 强依赖实验条件**（温度、pH、物种等）。

* 明确把 **pH、温度、organism** 编码成特征，与蛋白/底物表征一起输入，并用多任务结构同时预测 kcat 与 Km。([PubMed][3])
* 文中给了在同数据集上的对比提升（例如 kcat Pearson 0.808、Km Pearson 0.777，并对 DLKcat/UniKP 提升）。([PubMed][3])

当你写“为什么现有模型会有上限/噪声很大”时，MPEK 是非常好用的论据：**不建模条件变量 → 同一酶同一底物在不同实验条件下标签冲突**。

---

## 4) “变体/突变体动力学预测（定向进化语境）”的代表：EITLEM-Kinetics（Chem Catalysis 2024）

**EITLEM-Kinetics: A deep-learning framework for kinetic parameter prediction of mutant enzymes**（Chem Catalysis, 2024-09-19）([科学直通车][4])

它的定位和上面几篇不太一样：更偏“**突变体/多突变 → 动力学参数变化**”这一应用场景。

* 关键词是 **iterative transfer learning / ensemble / 低同源（<40%）** 的鲁棒性叙事。([科学直通车][4])
* 如果你做的是“突变导致的 kcat/Km 变化”或定向进化筛选，这篇比只做野生型大数据回归的工作更贴近任务本质。

---

## 5) “把结构信息更系统地引入（GNN/结构图）”：DEKP（Briefings in Bioinformatics 2025）

**DEKP: … based on pretrained models and graph neural networks**（Briefings in Bioinformatics, 2025）([OUP Academic][5])

它的典型卖点是：

* 认为纯序列会缺失关键 3D/口袋信息，因此引入结构图/GNN来提升泛化并缓解“训练-测试同源性差异导致掉点”。([OUP Academic][5])

如果你走的是“ESMFold/AlphaFold 结构 + pocket图 + substrate图”的路线，DEKP 是一条更接近你方向的已发表参照。

---

## 6) 对 DLKcat 的“稳定性/过拟合”质疑与改进：NNKcat（Briefings in Bioinformatics 2025）

**NNKcat**（Briefings in Bioinformatics, 2025-05-15）([OUP Academic][6])

它最值得看的点不是模型多复杂，而是它把社区里一直存在的痛点挑明了：

* 指出早期工作（包括 DLKcat）可能对数据划分/随机种子敏感，存在稳定性问题，并提出更可控的 protein/substrate processor 设计与“focused learning”。([OUP Academic][6])

在你写“现有 SOTA 的可靠性/可复现性问题”时，这篇很有用。

---

## 7) 预印本趋势：把“口袋-底物 3D 结合构象/对接”显式喂给模型（2025 bioRxiv）

这一类工作（比如 GraphKcat/KcatNet 等）共同信号是：**序列+SMILES 还不够 → 需要真实的 binding geometry**

* **GraphKcat / pocket-informed augmentation**（bioRxiv 2025-05）提出将酶-底物 3D 结合构象纳入图学习以提升预测。([biorxiv.org][7])
* **KcatNet**（bioRxiv 2025-03）强调几何深度学习做 genome-wide 预测。([biorxiv.org][8])

> 这条趋势和你正在做的“对接→提 pocket 原子→GNN”几乎同构：差别主要在于**他们如何规模化得到构象、如何评测冷启动、以及是否做不确定性/校准**。

---

# 我建议你把“sota”拆成 3 种含义（写论文/做实验会更稳）

1. **基准与可复用性 SOTA**：CatPred（数据+UQ+OOD）、CataPro（冷启动评测设计）
2. **把影响因子建模得更科学**：MPEK（条件变量 + 多任务）
3. **更贴近机理/结构**：DEKP（结构图/GNN）、以及 2025 预印本的 pocket-binding 构象路线

---

## 你如果要“最快搭一个像样的SOTA对照实验”，可以这么选对照组

* **经典基线**：DLKcat（2022 Nat Catalysis，仍然是很多后续论文的比较对象）([Nature][9])
* **强公开基线（PLM时代）**：UniKP（Nat. Commun. 2023）([Nature][10])
* **近期强框架**：CatPred + CataPro（2025）([Nature][1])
* **结构路线对照**：DEKP（2025）([OUP Academic][5])
* **条件变量对照**：MPEK（2024）([PubMed][3])

这样你的“related work + benchmark”就会非常完整：从 CNN/GNN 的 DLKcat → PLM 的 UniKP → 2025 的系统框架与冷启动评测 → 结构/口袋路线。

---

如果你愿意，我可以再进一步把上述论文按同一套维度给你做一个“**方法矩阵**”（输入模态、数据来源/清洗、split方式、指标、OOD设置、是否考虑温度pH、是否UQ、是否支持突变体），并顺带给出**你当前 pocket-GNN pipeline**最容易写出“创新点”的切入位置（通常在 *split设计 + 条件变量 + 口袋构象的可置信度/多构象集成 + 校准/UQ* 这几个点上）。

[1]: https://www.nature.com/articles/s41467-025-57215-9 "CatPred: a comprehensive framework for deep learning in vitro enzyme kinetic parameters | Nature Communications"
[2]: https://www.nature.com/articles/s41467-025-58038-4 "Robust enzyme discovery and engineering with deep learning using CataPro | Nature Communications"
[3]: https://pubmed.ncbi.nlm.nih.gov/39129365/ "MPEK: a multitask deep learning framework based on pretrained language models for enzymatic reaction kinetic parameters prediction - PubMed"
[4]: https://www.sciencedirect.com/science/article/pii/S2667109324002665?utm_source=chatgpt.com "EITLEM-Kinetics: A deep-learning framework for ..."
[5]: https://academic.oup.com/bib/article/26/2/bbaf187/8119324 "DEKP: a deep learning model for enzyme kinetic parameter prediction based on pretrained models and graph neural networks | Briefings in Bioinformatics | Oxford Academic"
[6]: https://academic.oup.com/bib/article/26/3/bbaf212/8131740 "NNKcat: deep neural network to predict catalytic constants (Kcat) by integrating protein sequence and substrate structure with enhanced data imbalance handling | Briefings in Bioinformatics | Oxford Academic"
[7]: https://www.biorxiv.org/content/10.1101/2025.05.18.654694v1.full-text?utm_source=chatgpt.com "Catalytic pocket-informed augmentation of enzyme kinetic ..."
[8]: https://www.biorxiv.org/content/10.1101/2025.03.09.642294v3?utm_source=chatgpt.com "KcatNet: A Geometric Deep Learning Framework for ..."
[9]: https://www.nature.com/articles/s41929-022-00798-z?utm_source=chatgpt.com "Deep learning-based kcat prediction enables improved ..."
[10]: https://www.nature.com/articles/s41467-023-44113-1?utm_source=chatgpt.com "UniKP: a unified framework for the prediction of enzyme ..."
