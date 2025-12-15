下面给你两样东西：

1. **可直接粘进论文 Action / Related Work 的完整综述段落（英文）**
2. **一张 literature timeline 图 + 一张方法对比表**（timeline 我已经给你生成好了）

---

## 1) 综述段落（可直接贴进 Action / Related Work）

### Related work: enzyme kinetics prediction

Predicting enzyme kinetic parameters such as turnover numbers (*k*cat) from sequence and structural information is challenging due to sparse measurements, heterogeneous assay conditions, and substantial experimental noise. Early deep-learning efforts such as **DLKcat** combined protein sequence features with small-molecule representations to predict *k*cat at scale, demonstrating that learned representations can capture enzyme–substrate-dependent trends under in-distribution evaluation. ([Nature][1]) More recent works increasingly emphasize robust evaluation and broader kinetic coverage: **UniKP** proposed a unified framework leveraging pretrained protein language models (pLMs) to predict multiple kinetic parameters, highlighting the utility of pLM-derived representations for enzymology tasks. ([PubMed][2]) In parallel, **CatPred** introduced a comprehensive framework for predicting *k*cat, *K*m, and *K*i, providing benchmark datasets with substantially expanded coverage and incorporating uncertainty estimates—an important step toward deployment in out-of-distribution settings. ([Nature][3]) Related efforts in the same direction include kinetic-parameter predictors such as **CataPro**, further illustrating the field-wide trend toward multimodal feature integration and broader parameterization. ([Nature][4])

### Related work: protein language models and multimodal fusion

A major driver of progress in enzyme property prediction has been the emergence of large-scale protein language models. The **ProtTrans** family (including T5-based variants often referred to as ProtT5) showed that self-supervised training on large protein corpora yields transferable representations for downstream function prediction. ([PubMed][5]) More recently, **ESM-2/ESMFold** demonstrated evolutionary-scale modeling and strong representational power, further motivating the use of frozen pLM embeddings as global context features in supervised tasks. ([科学杂志][6]) Beyond sequence-only models, there is increasing interest in **structure-aware** and **cross-modal** architectures. In molecular and pocket representation learning, **Uni-Mol** provided SE(3)-equivariant pretraining for both small molecules and protein pockets, enabling strong 3D inductive biases and transferable embeddings for binding-related tasks. ([ChemRxiv][7]) In adjacent areas such as protein design, inverse-folding models like **PiFold** leveraged graph-based structural encoders to efficiently generate sequences from backbone structures, indicating that geometric signals can be distilled effectively by GNN-style architectures. ([OpenReview][8])

### Related work: cross-attention and geometric GNNs for enzyme–substrate interaction reasoning

Cross-attention mechanisms and SE(3)-equivariant message passing have become increasingly popular for modeling enzyme–ligand or enzyme–substrate interactions. A recent example is **EZSpecificity**, which employs cross-attention–empowered SE(3)-equivariant GNNs for enzyme substrate specificity prediction, demonstrating that explicitly modeling interactions between modalities can improve performance and interpretability. ([Nature][9]) These developments collectively suggest a converging direction: combining **local 3D pocket geometry** (to capture stereochemical constraints and interaction patterns) with **global sequence-level evolutionary context** (to encode fold family, conservation, and mechanistic priors), while evaluating generalization under more stringent split protocols.

> 你可以把最后一句直接改写成“Therefore, we propose …”去引出你自己的模型。

---

## 2) Literature timeline 图（已生成，可直接用在论文/汇报）

[Download the timeline figure](sandbox:/mnt/data/literature_timeline_2020_2025.png)

（图中包含：ProtTrans/ProtT5、DLKcat、ESM-2/ESMFold、Uni-Mol、PiFold、UniKP、CatPred、EZSpecificity、CataPro）

---

## 3) 方法对比表（可直接作为论文表格雏形）

> 建议你把这一表放在 Related Work 末尾，或 Supplementary。

| Work                             |      Year | Task                             | Inputs                              | Core model idea                              | Notes for your paper                          |
| -------------------------------- | --------: | -------------------------------- | ----------------------------------- | -------------------------------------------- | --------------------------------------------- |
| ProtTrans / ProtT5 ([PubMed][5]) | 2020–2022 | General protein representation   | Sequence                            | Large pLM (T5/BERT-style)                    | 作为你 ESM/ProtT5 的“序列先验”背景                      |
| DLKcat ([Nature][1])             |      2022 | *k*cat prediction                | Sequence + substrate                | Deep model with learned representations      | 经典 kinetics baseline；强调评估设定差异                 |
| Uni-Mol ([OpenReview][10])       | 2022–2023 | 3D molecule/pocket rep. learning | 3D conformers / pockets             | SE(3)-equivariant pretraining                | 你的 pocket-3D 表征“同类方向”引用                       |
| PiFold ([OpenReview][8])         |      2023 | Inverse folding (design)         | Structure                           | Graph structural encoder + one-shot decoding | 证明结构图表征很强，但任务不同                               |
| ESM-2 / ESMFold ([科学杂志][6])      | 2022–2023 | pLM / structure prediction       | Sequence                            | Evolutionary-scale LM                        | 你做 “frozen ESM embedding + late fusion” 的核心依据 |
| UniKP ([PubMed][2])              |      2023 | *k*cat/*K*m/*k*cat/*K*m          | Sequence (+ substrate)              | pLM-based unified prediction                 | 重要 kinetics 相关 baseline（多参数）                  |
| CatPred ([Nature][3])            |      2025 | *k*cat/*K*m/*K*i + uncertainty   | (multimodal)                        | Framework + UQ + benchmarks                  | 强调 OOD 与不确定性（你可对齐 homology split）             |
| CataPro ([Nature][4])            |      2025 | Kinetics prediction              | pre-trained features + fingerprints | deep predictor                               | 可作为“同年 kinetics”补充引用                          |
| EZSpecificity ([Nature][9])      |      2025 | Enzyme specificity               | Sequence + structure                | cross-attention SE(3) GNN                    | 你做 cross-modal 融合/attention 的强背书              |

---

如果你愿意，我还能把这张“方法对比表”进一步升级成**更像 JCIM 风格的版本**：

* 额外加一列：**Split protocol / OOD evaluation 是否做了**
* 额外加一列：**是否提供 code/data**
  这两列对审稿人非常有杀伤力（尤其是你要突出 homology-aware split 的话）。

[1]: https://www.nature.com/articles/s41929-022-00798-z?utm_source=chatgpt.com "Deep learning-based kcat prediction enables improved ..."
[2]: https://pubmed.ncbi.nlm.nih.gov/38081905/?utm_source=chatgpt.com "UniKP: a unified framework for the prediction of enzyme ..."
[3]: https://www.nature.com/articles/s41467-025-57215-9?utm_source=chatgpt.com "CatPred: a comprehensive framework for deep learning in ..."
[4]: https://www.nature.com/articles/s41467-025-58038-4?utm_source=chatgpt.com "Robust enzyme discovery and engineering with deep ..."
[5]: https://pubmed.ncbi.nlm.nih.gov/34232869/?utm_source=chatgpt.com "ProtTrans: Toward Understanding the Language of Life ..."
[6]: https://www.science.org/doi/10.1126/science.ade2574?utm_source=chatgpt.com "Evolutionary-scale prediction of atomic-level protein ..."
[7]: https://chemrxiv.org/engage/chemrxiv/article-details/6402990d37e01856dc1d1581?utm_source=chatgpt.com "Uni-Mol: A Universal 3D Molecular Representation ..."
[8]: https://openreview.net/forum?id=oMsN9TYwJ0j&utm_source=chatgpt.com "PiFold: Toward effective and efficient protein inverse folding"
[9]: https://www.nature.com/articles/s41586-025-09697-2_reference.pdf?utm_source=chatgpt.com "Enzyme specificity prediction using cross attention graph ..."
[10]: https://openreview.net/forum?id=6K2RM6wVqKu&utm_source=chatgpt.com "Uni-Mol: A Universal 3D Molecular Representation ..."
