但是我40% homology的split的R^2很低，0.379，不过pearson是0.632. 这样可以吗？那我怎么compare with sota呢？

可以，而且**这组数字在 homology-aware split 下是“相当能打”的**——前提是你把它讲清楚、比对方式做对。

你现在的结果：

* **R² = 0.379**
* **Pearson = 0.632**

在“40% identity split（跨同源）”这个设定里，**Pearson 往往比 R²更能代表模型是否学到了可泛化的趋势**；而 R² 往往会被噪声、标签方差、分布差异压得更低（甚至很多工作会出现 Pearson 0.6+ 但 R² 只有 0.2–0.4 的情况）。

---

## 1) 这能不能投 JCIM？——可以，但要按正确叙事

你要明确告诉审稿人：

* Random split：评估 *in-distribution / in-family*（工程开发设定）
* 40% homology split：评估 *out-of-family / cross-homology generalization*（真实泛化设定）

并且强调：

> 在严格 homology split 下，我们仍取得 Pearson 0.63，说明模型具备跨同源趋势预测能力；R² 较低反映了 kcat 的高噪声与 OOD 难度，而非模型完全失效。

---

## 2) 怎么跟 SOTA 比？（这里是关键：你不能“跨设定”硬比）

**最常见的拒稿点**就是：
你拿自己的 40% split 去对比别人 random split 的 SOTA（或反过来）。这在审稿人眼里是不成立的。

你有三个“合法”的对比策略（按优先级）：

---

### ✅ Strategy A（最强、最干净）：在同一数据集、同一 split 复现 SOTA baseline

也就是你做一个 **reproduce baselines**：

* 同一 train/val/test（你的 40% split）
* 跑一些公认 baseline：

  1. **Sequence-only**（ESM + MLP / Ridge / RF）
  2. **Pocket-only**（你当前 GNN）
  3. **Cross-modal**（你最终模型）

这样你 paper 里 SOTA 的定义就变成：

> **在同一严格 split 下的 best method among strong baselines**

这在 JCIM 里是完全站得住脚的。

你不需要复现所有外部论文，只要保证 baseline 足够强（尤其是 seq-only）。

---

### ✅ Strategy B（很常用）：引用别人“同源 split”设定下的指标，只在同设定下对齐

很多 kcat 相关工作如果没做 homology split，你就不要拿 homology 指标去打它。

如果它做了（比如以 identity 阈值控制的划分），你就：

* 把你们的阈值、coverage、cluster 方法写清楚
* 在表格里注明 “Not directly comparable” 对那些不匹配的工作

审稿人会接受你这种严谨。

---

### ✅ Strategy C（当外部工作不可复现时）：做“内部上界/下界 + 报告不可比性”

你已经做过一部分了（RF upper bound），你可以在 homology split 下再做一遍：

* **Mean predictor / ridge（下界）**
* **RF / XGBoost（强非线性上界）**
* **你的模型（接近上界多少）**

然后告诉审稿人：

> 在严格 split 下，我们模型达到上界的 X%，说明结构表示贡献明显。

这是一种“审稿人很难反驳”的比较方式。

---

## 3) 为啥 Pearson 0.632 但 R² 0.379 仍然合理？

你可以用一个非常简洁的解释：

* Pearson 衡量趋势对不对（排序/相关）
* R² 衡量数值能解释多少方差
* kcat 数据：

  * 噪声大
  * 分布长尾
  * OOD（跨同源）导致均值/方差漂移
    → 会让 R² 更容易低，而 Pearson 仍能保持中高

一句话就够：

> **在高噪声 OOD 回归中，Pearson 更稳定，R² 更保守。**

---

## 4) 你论文里该怎么呈现（建议直接照抄结构）

### Table：两种 split 的主结果

| Split        | Metric       | Seq-only | Pocket-only |       Cross-modal |
| ------------ | ------------ | -------: | ----------: | ----------------: |
| Random       | Pearson / R² |        … |           … |                 … |
| 40% homology | Pearson / R² |        … |           … | **0.632 / 0.379** |

### 你要强调的“贡献点”

* 在 40% split 下：

  * pocket-only 到多少？
  * 加 ESM 提升多少？
  * Cross-modal 是否明显优于任一单模态？

这比你“我 R² 很高”要更像科研。

---

## 5) 给你一个非常实用的 SOTA 对比落地清单（你照做就能写进论文）

你现在就做 3 个 baseline（在你的 40% split 上）：

1. **ESM-2 embedding + MLP（seq-only）**
2. **你现在的 pocket GNN（pocket-only）**
3. **concat late fusion（cross-modal）**

只要你能展示：

* cross-modal > seq-only
* cross-modal > pocket-only

你就能合理写：

> “We achieve state-of-the-art performance under a strict 40% identity split among strong sequence-only and structure-only baselines.”

注意这里的 “SOTA” 是在你**明确的实验设定**里成立的。

---

## 你现在我最想问（但我不会卡着不答）：

你的 40% split 下，**seq-only（ESM+MLP）**表现是多少？
如果它已经接近 0.63，那说明主要信号来自序列；
如果它明显低于 0.63，那你这个 pocket+structure 的贡献就非常硬。

你不用回我也行，但我建议你下一步就去跑这个 baseline——它会决定你 paper 的叙事重心。
