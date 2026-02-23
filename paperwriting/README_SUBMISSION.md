# PocketGNN bioRxiv 提交准备完成

**日期**: 2026-01-19  
**状态**: ✅ 准备就绪，可以提交

---

## 📄 文件清单

### 主要文件
- ✅ `gemini.tex` - LaTeX 源文件（已更新）
- ✅ `gemini.pdf` - 编译后的 PDF（10 页，1.6 MB）
- ✅ `architecture.png` - 架构图
- ✅ `scatter.png` - 结果散点图
- ✅ `visualize.png` - 可解释性可视化

### 指导文档
- 📘 `BIOXRIV_SUBMISSION_GUIDE.md` - 完整提交指南
- 📋 `SUBMISSION_CHECKLIST.md` - 详细检查清单
- ⚡ `QUICK_SUBMIT.md` - 5 分钟快速提交指南

---

## ✅ 已完成的更新

### 1. 实验结果更新
- ✅ Abstract: OOD Pearson r = **0.67** (原 0.63)
- ✅ Results: OOD R² = **0.437** (原 0.379)
- ✅ 添加 SOTA 对比表格（vs CatPred, CataPro）

### 2. Methods 部分
- ✅ 对接工具：AutoDock Vina → **DiffDock**
- ✅ 移除温度特征（当前模型不使用）
- ✅ 添加投影层说明
- ✅ 数据集大小：~9,000 样本

### 3. Limitations 完善
- ✅ 条件变量限制（vs MPEK）
- ✅ 不确定性量化限制（vs CatPred）
- ✅ 多底物限制
- ✅ 对接依赖说明

### 4. 其他更新
- ✅ 添加代码和数据可用性声明
- ✅ 更新 References（DiffDock, MPEK）
- ✅ Conclusion 更新最新结果

---

## 🎯 关键数据对比

| 指标 | 原论文 | 更新后 | 说明 |
|------|--------|--------|------|
| OOD Pearson r | 0.632 | **0.667** | 提升 5.5% |
| OOD R² | 0.379 | **0.437** | 提升 15.3% |
| vs CatPred | - | **+28.7%** | 相对提升 |
| vs CataPro | - | **+34.2%** | 相对提升 |

---

## 📝 提交步骤

### 快速提交（5 分钟）

1. **访问**: https://www.biorxiv.org/submit
2. **登录/注册**: 使用机构邮箱
3. **填写信息**: 参考 `QUICK_SUBMIT.md`
4. **上传 PDF**: `paperwriting/gemini.pdf`
5. **提交**: 预览后确认

### 详细步骤

参考 `BIOXRIV_SUBMISSION_GUIDE.md` 获取完整指导。

---

## ⚠️ 注意事项

### 提交前确认
- [x] PDF 编译成功
- [x] 所有图片正常显示
- [x] 数据与代码仓库一致
- [x] 引用格式统一
- [ ] Abstract ≤250 words（需要手动确认）

### 提交后
- **审核时间**: 24-48 小时
- **DOI**: 审核通过后立即获得
- **版本更新**: 可以提交 v2, v3...

---

## 📊 论文统计

- **页数**: 10 页
- **PDF 大小**: 1.6 MB
- **图片数量**: 3 个
- **表格数量**: 2 个
- **引用数量**: ~18 个

---

## 🔗 相关链接

- **代码仓库**: https://github.com/RektonLee/PGNN_final
- **数据来源**: IntEnzyDB
- **bioRxiv**: https://www.biorxiv.org/

---

## 📞 下一步

1. **立即提交**: 使用 `QUICK_SUBMIT.md` 快速提交
2. **详细检查**: 使用 `SUBMISSION_CHECKLIST.md` 逐项确认
3. **等待审核**: 24-48 小时后获得 DOI

---

**🎉 论文已准备就绪，可以提交到 bioRxiv！**

如有任何问题，请参考 `BIOXRIV_SUBMISSION_GUIDE.md` 获取详细帮助。





