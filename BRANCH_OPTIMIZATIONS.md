# 分支优化说明（模型侧）

本仓库的压缩/解压是“同构推理 + 同步在线更新”的流程：压缩端和解压端必须在每一步得到一致的概率分布，并在相同的时机执行相同的参数更新，否则会导致解码失败。因此下面每个分支的改动都保持：

- 默认接口兼容（压缩/解压脚本不需要改参数即可运行）
- 在线更新仍然只发生在 `model_list[2] == 1`（启用学生模型）时

## opt-cache-rfs

**优化点**

- 将 Performer 随机特征（`favor+` 路径用到的 rfs）从“每次 forward 重新采样”改为“按 device 缓存后复用”。
- 修复 `valid_feature_type('favor+')` 会触发越界的问题，使 `favor+` 相关特征类型更可用。

**收益预期**

- 推理更快（避免重复采样与张量构造）
- 输出更稳定（减少额外随机性带来的方差）
- 更利于编码/解码一致性（随机特征不随步数变化）

**涉及文件**

- models_torch.py

## opt-gating-mixture

**优化点**

- 将原来的“全局常数融合权重（`self.weights`）”替换为“按样本自适应的门控融合（`self.gate`）”。
- 门控输入使用各模型 logits 的简单统计量（max/mean/var），输出每个样本的混合系数，然后对 logits 做加权求和。

**收益预期**

- 不同复杂度片段可自动选择更合适的教师/学生占比，通常能改善码率或减少在线更新压力
- 相比直接用 vocab 维度特征做门控，这种统计量门控开销更低

**涉及文件**

- models_torch.py

## opt-distill-kl

**优化点**

- 在在线更新阶段额外加入 teacher→student 的 KL 项，让学生更直接地拟合教师分布（不仅仅拟合 hard label）。
- 新增 `DistiallationNN.teacher_logits()`，在不启用梯度的情况下生成教师 logits（公共教师 + 私有教师的均值），用于构造 KL 的 target 分布。
- 在压缩与解压两侧的在线更新里同时加入该项，保持端到端一致。

**默认超参**

- 新增命令行参数 `--distill_alpha`，默认 `0.1`，用于控制 KL 项权重。

**收益预期**

- 更快把教师的“分布形状”迁移到学生，降低学生仅靠 hard label 学习带来的不稳定
- 在学生在线学习场景下，通常能改善收敛稳定性与压缩率（具体依赖数据集/模式）

**涉及文件**

- models_torch.py
- CPMKLC_S.py
- DCPMKLC_S.py

## 使用方式

```bash
git checkout opt-cache-rfs
# 或
git checkout opt-gating-mixture
# 或
git checkout opt-distill-kl
```

