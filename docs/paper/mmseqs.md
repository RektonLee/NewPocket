行，我给你一个**“40% identity split”的最小可行流程（MVP）**，目标是：**不大改你现有 pipeline**，只是在生成/加载 `.pt` 后做一次**基于同源聚类的分组切分**，保证 **同一同源簇不会跨 train/val/test**。

下面按“你现在最可能已有的东西”来设计：你每个图里至少有 `sample_id`（看你 Data 里就有），`pdb_id`，`y`。核心是：**让 sample_id ↔ 序列**对应起来。

---

## 0. 你需要准备的唯一输入

### 必须有：一个 FASTA 文件（每个 sample_id 一条序列）

格式像这样（header 里必须能提取 sample_id）：

```fasta
>kcat_000002
MKT...
>kcat_000003
...
```

如果你现在没有这个 fasta：通常你的原始数据（csv/tsv）里会有 sequence 列，或者能从 sample_id 对应的 Uniprot/序列文件拿到。MVP 先假设你能得到它。

---

## 1) 用 MMseqs2 或 CD-HIT 做 40% identity 聚类

我推荐 **MMseqs2**（快、稳、输出友好）；CD-HIT 也给你备选。

### 方案 A：MMseqs2（推荐）

```bash
mmseqs easy-cluster samples.fasta clusters tmp --min-seq-id 0.4 -c 0.8
```

输出里你会得到 `clusters_cluster.tsv`（或类似名字），一般是两列：

```text
representative_id    member_id
```

你可以把 `member_id` 当成 sample_id。

> `-c 0.8` 是覆盖度阈值（coverage），MVP 先用 0.8，后面可调。

### 方案 B：CD-HIT（备选）

```bash
cd-hit -i samples.fasta -o clustered.fasta -c 0.4 -n 2
```

然后用 `.clstr` 文件解析 cluster 与成员关系（略麻烦，但也可做）。

---

## 2) 把 cluster 变成 train/val/test（按 cluster 分组随机切）

**关键点**：切分单位是 **cluster_id**，不是样本、不是 pdb。

### 切分比例建议

* train/val/test = **80/10/10**（你现在习惯的方式）
* 或 85/10/5 也行（test 已经单独的话就按你的 test 定）

### MVP 规则

* 同一个 cluster 的所有 member（sample_id）只能落到一个 split 里

---

## 3) 把 split 映射回你的图数据 `.pt`

你的 `.pt` 里是 `List[Data]`，每个 Data 有 `sample_id`。做法：

* 先构建 `sample_id -> split` 的字典（来自 cluster split）
* 遍历 dataset：

  * `sid = data.sample_id`
  * 把 data 放进对应 split 的 list

然后保存：

* `kcat_train_hom40.pt`
* `kcat_val_hom40.pt`
* `kcat_test_hom40.pt`

> 你训练脚本不想改太多的话：让 `--dataset` 指向 train.pt，然后你代码里 val 从 train 再划分就不严谨了。更严谨是 train/val/test 三个都独立保存并加载（但这是后续增强，不影响 MVP）。

---

## 4) 给你一个“可直接跑”的 Python MVP 脚本

假设 MMseqs2 输出的成员表是 `clusters_cluster.tsv`（你根据实际文件名改一下）。

把下面存成：`scripts/make_homology_split.py`

```python
import argparse
import random
from collections import defaultdict
import torch

def read_mmseqs_tsv(tsv_path):
    # returns: member_id -> cluster_rep
    member_to_rep = {}
    with open(tsv_path, "r") as f:
        for line in f:
            rep, mem = line.strip().split("\t")[:2]
            member_to_rep[mem] = rep
    return member_to_rep

def make_cluster_ids(member_to_rep):
    # rep -> cluster_id (0..K-1)
    reps = sorted(set(member_to_rep.values()))
    rep_to_cid = {rep: i for i, rep in enumerate(reps)}
    member_to_cid = {m: rep_to_cid[rep] for m, rep in member_to_rep.items()}
    return member_to_cid, rep_to_cid

def split_clusters(rep_to_cid, seed=42, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    cids = list(rep_to_cid.values())
    random.Random(seed).shuffle(cids)
    n = len(cids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_c = set(cids[:n_train])
    val_c = set(cids[n_train:n_train+n_val])
    test_c = set(cids[n_train+n_val:])
    return train_c, val_c, test_c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="full dataset .pt (List[Data])")
    ap.add_argument("--mmseqs_tsv", required=True, help="mmseqs cluster tsv: rep<TAB>member")
    ap.add_argument("--out_prefix", required=True, help="output prefix, e.g. data/processed/kcat_full_hom40")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    args = ap.parse_args()

    dataset = torch.load(args.pt)  # List[Data]
    member_to_rep = read_mmseqs_tsv(args.mmseqs_tsv)
    member_to_cid, rep_to_cid = make_cluster_ids(member_to_rep)

    train_c, val_c, test_c = split_clusters(
        rep_to_cid,
        seed=args.seed,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )

    train, val, test, missing = [], [], [], 0
    for d in dataset:
        sid = getattr(d, "sample_id", None)
        if sid is None:
            raise ValueError("Data object missing sample_id; cannot do homology split.")
        if sid not in member_to_cid:
            missing += 1
            continue  # or assign to train; MVP: drop to avoid leakage
        cid = member_to_cid[sid]
        if cid in train_c:
            train.append(d)
        elif cid in val_c:
            val.append(d)
        else:
            test.append(d)

    print(f"Total graphs: {len(dataset)}")
    print(f"Missing sample_id in clustering table: {missing}")
    print(f"Train/Val/Test graphs: {len(train)}/{len(val)}/{len(test)}")

    torch.save(train, args.out_prefix + "_train.pt")
    torch.save(val, args.out_prefix + "_val.pt")
    torch.save(test, args.out_prefix + "_test.pt")

if __name__ == "__main__":
    main()
```

运行示例：

```bash
python scripts/make_homology_split.py \
  --pt data/processed/kcat_full.pt \
  --mmseqs_tsv clusters_cluster.tsv \
  --out_prefix data/processed/kcat_full_hom40 \
  --seed 42
```

---

## 5) 两个你一定要注意的“蛋白任务特有坑”

### 坑 1：同一个 sample_id 可能对应多个图（不同 pdb_id / pocket）

这是好事：**它们会自动被分到同一 split**（因为 split 是按 sample_id / cluster）。

### 坑 2：你的 `.pt` 里 sample_id 在聚类表里缺失

MVP 做法是我上面脚本里那样：**先 drop**，避免泄露。
更进阶做法：把缺失的 sample_id 单独记录，补齐 fasta 后再跑一次。

---

## 6) 你做完后，怎么证明“真的没泄露”？

最简单的 sanity check：

* 把 train/val/test 里的 `sample_id` 集合取交集，应该都为空
* 再把 cluster_id 的集合取交集，也应该都为空

---

如果你把 `samples.fasta` 的 header 形式（里面 sample_id 怎么写）和 mmseqs 输出文件名贴一小段，我可以把上面的脚本里“解析 sample_id”的部分给你定制到完全贴合你数据（避免你因为一个 header 格式卡住）。
