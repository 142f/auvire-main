# D1 文献卡片与 Gap 矩阵 —— 音视频深伪「时序定位（segment-level TFL）」方向

> 项目：AuViRe 复现与改进（基线：Koutlis & Papadopoulos, *AuViRe: Audio-visual Speech Representation Reconstruction for Deepfake Temporal Localization*, **WACV 2026**, arXiv:2511.18993）
> 产出日期：2026-11-09　|　文档定位：论文 Related Work + Gap 论证的原料库
> 覆盖：12 张卡片（含 2 处替代说明）、1 张 Gap 矩阵、5 条候选缺口、1 张横向对照表、1 份未核实清单

---

## 0. 核实口径（读之前请先看）

**标记约定**

| 标记 | 含义 |
|---|---|
| 【核实】 | 我在本轮检索中直接读到了原文/官方表格/官方 README 的该数字或该表述 |
| 【未核实】 | 我未能拿到可核验来源（原文被截断、无公开全文、只有二手转述），**不做任何推测填充** |
| 【本项目实测】 | 来自本仓库产物（results/*.json、ckpt 日志、src 代码），非论文原文数字 |

**本轮检索的实际限制（必须写进论文的"数据来源"脚注，避免审稿人质疑）**

1. `web_search` 工具在本会话**不可用**（搜索端点返回 HTTP 401 认证失败），全部检索改由 `web_fetch` 直连完成：arXiv abs/HTML、`export.arxiv.org` API、`api.crossref.org`、`raw.githubusercontent.com`（官方仓库 README）、`openaccess.thecvf.com`。OpenAlex / Semantic Scholar / DuckDuckGo / r.jina.ai 均被限流或拒绝。
2. arXiv 的 HTML 单次抓取有 **100,000 字符上限**（响应 `truncated=true`），而 LaTeXML 页面被 `<math>` 标签严重稀释，正文常常只读到**方法节中段**，**实验表与 Limitations 段多数读不到**。因此：**表格数字优先取自"官方仓库 README / Codabench 提交结果 / 本仓库复现结果"**，并在卡片中注明来源；读不到的一律写【未核实】。
3. 因此本文件对"报告数字"的处理原则是：**能溯源到官方表格的写死，溯不到的一律留白**，绝不用"约/大致/我记得"来填。

**卡片清单与两处替代（先说清楚，避免与你的清单错位）**

| # | 你给的条目 | 本文件实际写的条目 | 处置 |
|---|---|---|---|
| 1 | AuViRe (2511.18993) | AuViRe | 直接对应 |
| 2 | LAV-DF: "Lips Are All You Need for Deepfake Detection" | **LAV-DF 数据集论文**（"Do You Really Mean That? ...", arXiv:2204.06228, DICTA 2022） | **标题无法核实**：arXiv 全文检索 `abs:"lips are all you need"` 返回 0 条，Crossref 亦无匹配。LAV-DF 的可核实出处是 2204.06228（DICTA 2022）与 2305.01979（CVIU），故按此二文写卡片 2 与卡片 4 |
| 3 | AV-Deepfake1M (+ AV-LSTM) | AV-Deepfake1M | 直接对应（AV-LSTM 细节【未核实】，见卡片 3 第 7 条） |
| 4 | BA-TFD / BA-TFD+ "A Bi-modal Audio-Visual Transformer" | **BA-TFD / BA-TFD+（Boundary Aware Temporal Forgery Detection）** | **标题需更正**：BA-TFD 的官方全称是 *Boundary Aware Temporal Forgery Detection*（见 2305.01979 摘要原文），不是 "Bi-modal Audio-Visual Transformer"；后者这一措辞我未在任何可核实来源中找到【未核实】 |
| 5 | UMMAFormer | UMMAFormer (2308.14395, ACM MM 2023) | 直接对应 |
| 6 | TALL (ICCV 2023) | TALL (2307.07494) | 直接对应 |
| 7 | CFAD | **替代：HBMNet（Localizing Audio-Visual Deepfakes via Hierarchical Boundary Modeling, arXiv:2508.02000）** | **CFAD 无法核实**：arXiv `ti:CFAD`、`all:"CFAD" AND all:"deepfake"` 均无相关命中；Crossref 无匹配。备选替代（"Temporal Forgery Localization survey"、"Exploiting Complementary Audio-visual Competitive Cues for Deepfake Detection"）**同样无法核实**（后者 Crossref 无记录、arXiv 无记录）。最终选 HBMNet：它是 2025 年 AV-TFL 方法，且**摘要中明确与 BA-TFD、UMMAFormer 对比**，对本文的"时序定位骨架族"论证最有用 |
| 8 | ActionFormer (ECCV 2022) | ActionFormer (2202.07925) | 直接对应 |
| 9 | DIMODIF | **DiMoDif**（arXiv:2411.10193，作者与 AuViRe 同一组） | 直接对应（写法更正为 DiMoDif） |
| 10 | AVFF (CVPR 2024) | AVFF (2406.02951) | 直接对应（**注意：arXiv 号不是 2404.xxxx**） |
| 11 | MDS | **MDS = Modality Dissonance Score 网络**（Chugh et al., *Not made for each other*, ACM MM 2020, arXiv:2005.14405） | **已解出歧义**：在 AV-DFD 文献里 "MDS" 就是 Chugh 2020 提出的 *Modality Dissonance Score*（AVFF 与 BA-TFD 论文的 Related Work 均如此引用）。它不是 "MDS-Net"，也不是 "Multi-Dimensional Scaling"。**无需替代** |
| 12 | 一篇 AV 不一致性论文 | **AVoiD-DF**（IEEE TIFS 2023） | 元数据【核实】、技术细节【未核实】（IEEE 全文不可达，arXiv 无版本） |

> 一句话结论：**"CFAD" 与 "Lips Are All You Need" 这两个标题在你给的清单里无法在公开检索中落地**，本文件用可核实论文顶上并逐条标注；**"MDS" 反而可以精确落地为 Chugh 2020 的 Modality Dissonance Score**。

---

## 1. 文献卡片（12 张）

### 卡片 1 ｜AuViRe（本项目的基线）
**AuViRe: Audio-visual Speech Representation Reconstruction for Deepfake Temporal Localization** — Koutlis & Papadopoulos，WACV 2026，arXiv:2511.18993【核实】；代码 mever-team/auvire

1. **任务定义**：输入**整段**音视频（视觉流 `Φ`：`t` 帧；音频流 `h`：`s` 采样点），输出**逐帧**伪造概率 `p_hat ∈ R^(t×l)` 与**逐帧边界** `b_hat ∈ R^(t×l×2)`（起止时间，单位秒，仅对伪造帧有效）。核心任务是 **TFL（Temporal Forgery Localization，段级时序定位）**；**DFD（视频级真/假）是"涌现"出来的副产品**：视频级分数 `y_hat = max_j ρ_j`（soft-NMS 后最高分段置信度），标签 `y = OR_τ p^τ`。即：**不是分类模型，而是"帧级二分类头 + 边界回归头"的稠密预测模型**。
2. **输入表示**：**AV-HuBERT（Base / LRS3 / iteration-4 / 不微调）冻结特征**。视觉与音频**分别单独**过 backbone（另一模态喂 `null`，即 `x_v = B(Φ, null)`、`x_a = B(null, h)`），**刻意避免模态早期融合**，以保留跨模态差异；`x_v, x_a ∈ R^(t×d)`（`d=768`，AV-HuBERT Base 隐层维；本地 `features.npz` 实测为 `(T,768)` float32，`T≤495`【本项目实测】）。原始像素/波形只用于特征提取，训练与评测都在特征域进行（**特征必须离线预提**，见仓库 README 的 "AuViRe expects AVHubert features as input"）【核实】。
3. **时序建模方式**：**纯 1D 卷积（无 Transformer 主干）**，三段：
   - 重建器 `R`：pre-proj（1D conv, `κ=3`）→ 下采样（`κ=15, σ=2`）×`l_down` → 上采样反卷积 ×`l_up` → post-proj；同一 `R` 交替承担 `a→v`、`v→v`、`a→a` 三种重建；
   - 残差构造 `γ(ζ,η) = x_hat_(ζ,η) − x_η`，三路拼接成 `3d` 维输入；
   - 重建残差编码器 `E`：若干 stride-1 卷积（保时间分辨率）+ 若干 stride-2 卷积（降采样），**顶端加 FPN**，得到 `l` 层多尺度伪造特征 `f ∈ R^(t×l×q)`；
   - 两个头：分类头 `C`（`κ=3`，`d_out=1`）、回归头 `G`（`d_out=2`）；
   - **后处理用 SoftNMS**（重加权重叠段）。
   - 消融里给了 CNN vs Transformer 主干对比（`results/ablation/model_type/*transformer*`）【本项目实测】。
4. **监督信号与标签粒度**：**混合粒度**——`y ∈ R^(t×3)` = 帧级二值 `p ∈ {0,1}^t`（**帧级整数标签**）+ 帧内边界 `b ∈ [0,Δ]^(t×2)`（**连续秒**，仅伪造帧有效）。损失：`L_loc` = focal（逐帧分类）+ 指示函数 `1_{p^τ}` × DIoU（边界回归），按 `Σ p^τ` 归一化；`L_rec` = **只在"整段为真"的视频上生效**（`AND_τ ¬p^τ` 指示函数）的重建 MAE；`L = (L_loc + L_rec)/2`。**注意**：重建正则的生效条件是"全真视频"，这在"整段都被篡改"的样本上等于没有正则项（见第 7 条）。
5. **评测口径**：**AP@{IoU 阈值} + AR@{top-N 提案数}**，在 **SoftNMS 之后**计算；视频级 DFD 由 `max ρ_j` 导出后算 Acc/AUC/AP。仓库实现（`src/eval.py`）的阈值集合：AP 用 `lavdf:[0.5,0.75,0.95]`、`avdeepfake1m:[0.5,0.75,0.9,0.95]`；AR 用 `[0.5,0.55,...,0.95]` 十档；DFD 的 Acc 用 **固定 0.5 阈值**二值化（`accuracy_score(y_true, y_score>0.5)`，`src/eval.py:161`）【本项目实测】。AV-Deepfake1M **test 集标签不公开**，只能通过 Codabench 提交（仓库 `scripts/predict.py` + zip 提交）【核实】。
6. **报告数字**：
   - **论文口径（摘要/README）**：LAV-DF **+8.9 AP@0.95**、AV-Deepfake1M **+9.6 AP@0.5**、in-the-wild **+5.1 AUC**【核实】。
   - **LAV-DF test（同库）**【本项目实测，`results/test/task_tfl_training_on_lavdf.json`】：AP@0.5 **98.91**、AP@0.75 **96.03**、AP@0.9 **72.10**、AP@0.95 **46.52**、AR@100 **94.93**、AR@50 94.56、AR@30 94.28、AR@20 93.99、AR@10 93.28、AR@5 **92.16**。
   - **AV-Deepfake1M（验证集）**【本项目实测】：AP@0.5 **98.81**、AP@0.75 **92.17**、AP@0.9 45.56、AP@0.95 **13.17**、AR@100 88.06、AR@5 86.11。
   - **跨库（LAV-DF→AV-Deepfake1M）**【本项目实测】：AP@0.5 **16.05**、@0.75 6.68、@0.9 0.65、@0.95 0.05、AR@100 33.00。
   - **跨库（AV-Deepfake1M→LAV-DF）**【本项目实测】：AP@0.5 **53.76**、@0.75 43.26、@0.9 13.92、@0.95 0.85、AR@100 78.04。
   - **Codabench（AV-Deepfake1M test，官方提交结果）**【本项目实测，`results/avdeepfake1m_test_predictions/*/metrics.json`】：1M 训练 → AP@0.5 **0.9655**、@0.75 0.8928、@0.9 0.4290、@0.95 **0.1172**、AR@5 0.8382、DFD **AUC 0.9978**；LAV-DF 训练 → AP@0.5 **0.1470**、AP@0.95 0.00047、DFD AUC **0.6570**。
   - **DFD（视频级）**【本项目实测】：LAV-DF 上 Acc **73.54** / AP 99.98 / AUC **99.94**；AV-Deepfake1M 上 Acc **74.97** / AP 99.997 / AUC **99.99**；LAV-DF 模型迁到 1M 时 AUC 掉到 **65.71**。
   - 计算效率（参数量/FLOPs/吞吐）**未核实**（正文 §5.6 Computational Efficiency 落在抓取截断之外）。
7. **没解决的问题 / 自认的局限 / 悄悄回避的设定**：
   - **只覆盖"语音在场 + 单说话人 + 正脸"场景**：整条管线建立在 AV-HuBERT 语音表征上，论文的"real-world 实现"明确要求 "presence of both visual and audio streams … and **talking human presence**"【核实】——即**非语音段（纯音乐、画外音、无脸镜头）没有任何机制**，而真实取证视频里这类片段占多数。
   - **固定 512 帧窗口**（"maximum sequence length `t`=512, zero-padding smaller sequences"）【核实】，长视频只能切窗；**论文未给"切窗→拼接→跨窗去重"的推理协议**，而 in-the-wild 中位时长 1 分钟、最长 11.3 分钟【核实】——**长视频评测口径是隐性缺口**。
   - **跨库断崖**：同库 AP@0.5 98.81 / 跨库 16.05（LAV-DF→1M）；Codabench 上 LAV-DF 训练只有 0.147 AP@0.5、0.657 AUC【本项目实测】。论文主表并未以"跨库为主指标"，而是把跨库放在附录/泛化小节。
   - **"定位训练"产出的 DFD 分数不可校准**：AP 99.98 但**固定 0.5 阈值**的 Acc 只有 73.54【本项目实测】——说明 `max ρ_j` 的**绝对数值尺度没有语义**；论文附录确有 "D Threshold Calibration" 一节标题【核实】，但正文口径未核实。
   - **重建残差正则只对"全真视频"生效**（`AND_τ ¬p^τ`）：对"整段被换脸"的样本退化为无正则，实际把"部分篡改"设成了默认场景。
   - **强依赖 AV-HuBERT 冻结点**：骨干不参与训练，重建目标就是它的 768 维表征；**"重建得准"与"伪造可分"之间的张力**（backbone 越强，残差越接近噪声）未被讨论；论文只做了 Ma et al. 2022 特征的骨干替换消融【本项目实测，`results/ablation/backbone/`】。
   - **SoftNMS 超参（σ, t1, t2）与 AP 的耦合**未做敏感性分析（仓库中为固定值，`src/post_process.py`）【本项目实测】。
   - 论文正文的 Limitations 段落**未核实**（HTML 抓取被 100k 字符截断），以上第 1–6 条均从**可核实的方法/实验设置**推出，不是转述作者的自我批评。

---

### 卡片 2 ｜LAV-DF（数据集 + BA-TFD 基线）
**"Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization"** — Cai et al.，**DICTA 2022**，arXiv:2204.06228【核实】

1. **任务定义**：提出**内容驱动（content-driven）**的音视频深伪数据集 LAV-DF，并给出**面向 TFL 的基线方法 BA-TFD**。任务同时报 **TFL（段级定位：起止 + 真/假）** 与 **DFD（视频级二分类）**；关键设定是"只改一小段、把整句语义/情感极性反转"，因此**定位才是主任务**。
2. **输入表示**：**原始像素 + 原始音频**。视频编码器 `E_v` 吃整段 `V ∈ R^(C×T×H×W)`（**3DCNN，4 个 block**）输出帧级特征 `F_v ∈ R^(C_f×T)`；音频编码器 `E_a` 吃 **log-mel 谱** `A' ∈ R^(F_m×T_a)`（**2DCNN**）输出音频特征，并**在时间轴上与视频帧级特征对齐**【核实，§IV-A】。
3. **时序建模方式**：**3D CNN（时空卷积）+ 帧级 1D 卷积分类器 + 边界图（boundary map）回归 + 多模态融合**。融合模块用 1D 卷积预测音/视权重后做**加权平均**；两个边界匹配层 `B_v, B_a` 在特征上预测边界图；**没有 proposal 生成、没有 Transformer、没有显式段排序**。
4. **监督信号与标签粒度**：**帧级二值标签 + 边界图（帧级软标签）**。① 帧分类损失（逐帧交叉熵，且对模态做掩码：`Y_m = η_m·Y + (1−η_m)·Y_0`，`η_m` 表示该模态是否被改）；② 边界匹配损失（对预测边界图 `M` 做 **MSE**，同理做模态掩码）；③ 对比损失（拉近/推远音视特征）【核实，§IV-B】。**标签是"帧索引"级别的整数区间**，这是后续所有 @0.95 指标塌陷的根源之一（见卡片 1 第 4 条与 §3 缺口 G1）。
5. **评测口径**：**AP@{0.5, 0.75, 0.95} + AR@{10, 20, 50, 100}**，在 LAV-DF test 上报告（官方 README 表头即为这 7 列）【核实】。划分：train 78,703 / val 31,501 / test 26,100（AuViRe 论文转述的同一组数字）【核实】。
6. **报告数字**（LAV-DF test，官方仓库 README 表格）【核实】：

   | 方法 | AP@0.5 | AP@0.75 | AP@0.95 | AR@100 | AR@50 | AR@20 | AR@10 |
   |---|---|---|---|---|---|---|---|
   | BA-TFD | 79.15 | 38.57 | **0.24** | 67.03 | 64.18 | 60.89 | 58.51 |
   | BA-TFD+ | 96.30 | 84.96 | **4.44** | 81.62 | 80.48 | 79.40 | 78.75 |

   另有**人类基线**（AV-Deepfake1M 论文 Table 5 的用户研究，同一评测口径）：LAV-DF 上 Acc 84.03、AP@0.1 36.80、AP@0.5 **14.17**、AR@1 10.04【核实】。
7. **没解决的问题 / 局限**：
   - **@0.95 基本为 0**（BA-TFD 0.24 / BA-TFD+ 4.44）：**边界精度从头到尾没被当成目标**，只靠边界图的 MSE 间接学；论文的卖点是 "strong performance for TFL"，但高分位指标从未进入主叙事。
   - **数据是合成流水线**（内容改写 + TTS/换脸），未讨论**跨数据集泛化**（且该数据集只有"单说话人、正脸、语音"场景）。
   - **AP/AR 单点报告**：没有 proposal 数量-召回曲线、没有置信度校准、没有阈值迁移实验。
   - **"内容驱动"这一设定本身就是回避**：真实深伪更多是"整段换脸/换声"，而该数据集刻意构造"只改一小段且语义反转"，使任务在语义上可解——**换成"无意义片段篡改"时假设失效**。
   - 人类 AP@0.5 只有 14.17【核实】：说明**"段边界"这个标签本身对人类都不可辨**，标签噪声问题被数据集设计掩盖。

---

### 卡片 3 ｜AV-Deepfake1M（数据集 + benchmark；含 AV-LSTM 基线）
**"AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset"** — Cai et al.，**ACM MM 2024**，arXiv:2311.15308【核实】

1. **任务定义**：构建 **>2K 主体、>1M 视频**的 LLM 驱动音视频深伪数据集，覆盖 **(i) 视觉篡改、(ii) 音频篡改、(iii) 音视频同时篡改** 三种内容驱动局部篡改；任务 = **TFL（段级定位）+ DFD（视频级）** 双任务 benchmark。
2. **输入表示**：**原始视频 + 原始音频**（数据集层面不规定表征）；benchmark 中各方法自带前端：Meso4 / MesoInception4 / EfficientViT（视觉分类器）、TriDet、ActionFormer + VideoMAEv2 / InternVideo（时序定位骨架）、BA-TFD / BA-TFD+ / UMMAFormer（TFL 专用）、PyAnnote（纯音频说话人日志）【核实，官方 README 表格】。
3. **时序建模方式**：数据集本身不设时序模块；**基线 AV-LSTM 是音视频 LSTM 序列模型**（帧级输出 + 段聚合）。**其网络结构、隐藏维、损失、以及具体 AP 数字均【未核实】**——论文 benchmark 表落在 100k 抓取截断之外，官方 README 的 TFL 表也**没有列 AV-LSTM**（只列了上表那些方法）。benchmark 里的时序范式实际分成两派：**proposal/边界图派（BA-TFD、UMMAFormer）** 与 **backbone 迁移派（ActionFormer+VideoMAE/InternVideo）**。
4. **监督信号与标签粒度**：**段级（秒级起止时间）+ 视频级标签**。元数据 `metadata.json` 给出 `fake_segments`（start/end）与 `modify_type ∈ {real, visual_modified, audio_modified, both_modified}`，DFD 评测按 `modify_type` 分组【核实，README §Data】。**没有帧级掩码标签**。
5. **评测口径**：TFL 用 **AP@{0.5, 0.75, 0.9, 0.95} + AR@{50, 30, 20, 10, 5}**；1M-Deepfakes Challenge 的官方综合分为 `S = (Σ_{IoU∈{0.5,0.75,0.9,0.95}} AP@IoU)/8 + (Σ_{N∈{50,30,20,10,5}} AR@N)/10`【核实，挑战赛论文 Eq.】；DFD 用 **AUC**。**test 标签不公开，只能 Codabench 提交**（每队提交次数受限）。
6. **报告数字**（官方 README TFL 表，AP@0.5 / @0.75 / @0.9 / @0.95，AR@50/20/10/5）【核实】：

   | 方法 | AP@0.5 | AP@0.75 | AP@0.9 | AP@0.95 | AR@50 | AR@20 | AR@10 | AR@5 |
   |---|---|---|---|---|---|---|---|---|
   | PyAnnote | 0.03 | 0.00 | 0.00 | 0.00 | 0.67 | 0.67 | 0.67 | 0.67 |
   | Meso4 | 9.86 | 6.05 | 2.22 | 0.59 | 38.92 | 38.81 | 36.47 | 26.91 |
   | MesoInception4 | 8.50 | 5.16 | 1.89 | 0.50 | 39.27 | 39.00 | 35.78 | 24.59 |
   | EfficientViT | 14.71 | 2.42 | 0.13 | 0.01 | 27.04 | 26.43 | 23.90 | 20.31 |
   | TriDet + VideoMAEv2 | 21.67 | 5.83 | 0.54 | 0.06 | 20.27 | 20.12 | 19.50 | 18.18 |
   | TriDet + InternVideo | 29.66 | 9.02 | 0.79 | 0.09 | 24.08 | 23.96 | 23.50 | 22.55 |
   | ActionFormer + VideoMAEv2 | 20.24 | 5.73 | 0.57 | 0.07 | 19.97 | 19.81 | 19.11 | 17.80 |
   | ActionFormer + InternVideo | 36.08 | 12.01 | 1.23 | 0.16 | 27.11 | 27.00 | 26.60 | 25.80 |
   | BA-TFD | 37.37 | 6.34 | 0.19 | 0.02 | 45.55 | 35.95 | 30.66 | 26.82 |
   | BA-TFD+ | 44.42 | 13.64 | 0.48 | 0.03 | 48.86 | 40.37 | 34.67 | 29.88 |
   | UMMAFormer | **51.64** | **28.07** | **7.65** | **1.58** | 44.07 | 43.45 | 42.09 | 40.27 |

   人类基线（Table 5 用户研究）：**Acc 68.64、AP@0.1 15.32、AP@0.5 1.92、AR@1 2.54**【核实】。数据规模：AuViRe 转述为 train 746,180 / val 57,340 / test 343,240（test 无标签）【核实】。
7. **没解决的问题 / 局限**：
   - **整张表 AP@0.95 ≤ 1.58**：数据集层面的"边界不可辨"——即**用当前标注，@0.95 这个指标几乎不可能被显著改善**，而论文把这一现象归因于"方法不够强"。
   - **人类 AP@0.5 只有 1.92**【核实】：人类都定不出边界，说明 **"段"的定义本身是生成流水线的副产物**（拼接点），而不是感知上可辨的事件边界 → **标签噪声/粒度问题是结构性的**。
   - **AV-LSTM 基线的可复现性缺口**：论文正文给了它，但官方 README 的复现表里没有它，代码仓也无其实现（仓库仅提供数据与评测 API）【核实：GitHub 目录只有 `python/`、`src/`、`assets/` 等，无 baseline 训练代码】。
   - **test 标签封闭 + Codabench 提交限额**：导致**任何"评测协议"层面的改进（如阈值校准、NMS 变体）都无法被第三方复核**。
   - **合成痕迹**：VITS（721 个说话人模型）+ Whisper 转写，~3000 GPU 小时生成；**未讨论 TTS 系统指纹**与"检测器学到的是 TTS 伪影而非跨模态不一致"这一混淆变量（这是 AV-DFD 领域的通病，该文亦未做控制实验）。

---

### 卡片 4 ｜BA-TFD / BA-TFD+（Boundary Aware Temporal Forgery Detection）
**"Glitch in the Matrix: A Large Scale Benchmark for Content Driven Audio-Visual Forgery Detection and Localization"** — Cai et al.，CVIU（审稿中版本），arXiv:2305.01979【核实】
> 说明：该文即 LAV-DF 数据集 + **BA-TFD** 基线的正式版，**BA-TFD+** 为同一文的改进版（摘要原文："We further improve (i.e. BA-TFD+) the baseline method by replacing the backbone with a Multiscale Vision Transformer"）【核实】。你清单里的标题 "A Bi-modal Audio-Visual Transformer for Deepfake Detection" **无法核实**。

1. **任务定义**：**TFL + DFD 双任务**。输入整段音视频，输出**伪造段的起止时间**（+ 视频级真/假）；强调"只改小片段、其余与原始完全一致"的 content-driven 局部篡改。
2. **输入表示**：**原始视频帧 + 原始音频**（不依赖预训练语音表征）；BA-TFD 用 **3D CNN** 提视觉时空特征，音频走谱图分支；BA-TFD+ 把视觉骨干换成 **Multiscale Vision Transformer（MViT）**。
3. **时序建模方式**：**边界图（boundary map）回归 + 多模态边界匹配**。`M`、`M_v`、`M_a` 三张边界图分别监督，再做**多模态边界匹配损失**（把音/视边界图与融合边界图对齐）；本质是"逐帧边界概率 + 融合"，**没有 proposal 候选生成、没有段级排序、没有 NMS 设计**。
4. **监督信号与标签粒度**：**四项损失联合**：对比损失（contrastive）+ 帧分类（frame classification）+ 边界匹配（boundary matching）+ **多模态边界匹配（multimodal boundary matching）**【核实，摘要】。标签粒度 = **帧级二值 + 帧级边界图（软标签）**，边界图由 GT 段按文献 [52] 的流程生成。
5. **评测口径**：**AP@{0.5, 0.75, 0.95} + AR@{10, 20, 50, 100}**（LAV-DF 官方 README 的列即此）；AV-Deepfake1M 上另加 @0.9 与 AR@5。
6. **报告数字**：
   - LAV-DF test：BA-TFD **79.15 / 38.57 / 0.24**（AP@0.5/0.75/0.95），AR@100 **67.03**；BA-TFD+ **96.30 / 84.96 / 4.44**，AR@100 **81.62**【核实，官方 README】。
   - AV-Deepfake1M：BA-TFD **37.37 / 6.34 / 0.19 / 0.02**（AP@0.5/0.75/0.9/0.95）；BA-TFD+ **44.42 / 13.64 / 0.48 / 0.03**【核实，官方 README】。
7. **没解决的问题 / 局限**：
   - **边界图范式的天花板在 @0.95**：BA-TFD+ 从 BA-TFD 的 0.24 提到 4.44，仍是"个位数"；**换成 MViT 骨干只解决"特征"，没解决"边界定位的监督形式"**。
   - **没有"段"的输出**：边界图阈值化即可得段，但**阈值怎么选、段怎么去重、置信度怎么排序**论文未给；因此它天然无法与 proposal-based（UMMAFormer）在同一"AR@N"意义下公平比较（AP 可以，AR 的定义其实不对等）。
   - **跨数据集只在 LAV-DF/AV-Deepfake1M 之间**，且**1M 上 AP@0.95 只有 0.03**，说明边界图在"更短、更细"的篡改上失效。
   - **3D CNN 的计算成本**未与"效率"对照（TALL/ActionFormer 那类工作会明确报 FLOPs），BA-TFD+ 换 MViT 后成本上升多少【未核实】。
   - **"多模态边界匹配"的隐含假设**：音、视篡改段**边界完全对齐**；对"只改音频"或"音视错位"样本，该损失会把对齐误差当噪声学进去（AV-Deepfake1M 里 audio-only/visual-only 样本占多数，正好踩这个坑）。

---

### 卡片 5 ｜UMMAFormer
**"UMMAFormer: A Universal Multimodal-adaptive Transformer Framework for Temporal Forgery Localization"** — Zhang et al.，**ACM MM 2023**，arXiv:2308.14395【核实】

1. **任务定义**：**通用 TFL**：输入未裁剪的**视觉-only / 音频-only / 音视频联合**序列，输出伪造段集合 `S = {(t_{n,s}, t_{n,e}, s_n)}`（起止 + 置信度）。明确与 TAL（时序动作定位）区分：TFL 的"篡改"无类别语义、段更短、真样本更多。
2. **输入表示**：**预提特征**——视觉用 **TSN**（mmaction2 提取），音频用 **BYOL-A**；两模态各自编码后做自适应融合（README 提供 TSN 特征与 BYOL-A 流程）【核实】。
3. **时序建模方式**：**Transformer 主干 + 重建式异常注意力 + 跨注意力特征金字塔 + 逐位置预测头**：
   - **TFAA（Temporal Feature Abnormal Attention）**：基于**时序特征重建**（CRATrans，交叉重建注意力）判断"异常状态"，用重建残差强化时序差异；
   - **PCA-FPN（Parallel Cross-Attention FPN）**：并行 + 下采样双分支同时融合多尺度，用跨注意力（CA）交互，专门针对**极短伪造段**；
   - 顶端 **分类头 + 回归头**（逐位置，直接沿用 ActionFormer 的解码与损失设定："Other settings are directly adopted from ActionFormer"）【核实】。
4. **监督信号与标签粒度**：**逐点（per-timestamp）二分类 + 边界回归**："the label of forgery segments is set to 1 and the rest is set to 0"；损失 `L = L_cls + λ_reg·L_reg + λ_rec·L_rec + λ_scls·L_scls`（分类 + 回归 + 重建 + 样本级 focal）【核实】。粒度 = **帧/点级标签 + 段边界**。
5. **评测口径**：**AP@{0.5, 0.75, 0.95} + AR@{10, 20, 50, 100}**；另提供 `tools/test_miou.py` 按子集算 IoU（Psynd 的 test_cellular / test_landline 分列）【核实，README】。
6. **报告数字**（官方 README 复现表）【核实】：

   | 数据集 | 模态 | AP@0.5 | AP@0.75 | AP@0.95 | AR@10 | AR@20 | AR@50 | AR@100 |
   |---|---|---|---|---|---|---|---|---|
   | LAV-DF | V | 97.30 | 92.96 | **25.68** | 90.19 | 90.85 | 91.14 | 91.18 |
   | LAV-DF | V+A | **98.83** | **95.54** | **37.61** | 92.10 | 92.42 | 92.47 | 92.48 |
   | LAV-DF Subset | V | 98.83 | 95.95 | 30.11 | 92.32 | 92.65 | 92.74 | 92.75 |
   | LAV-DF Subset | V+A | 98.54 | 94.30 | 37.52 | 91.61 | 91.97 | 92.06 | 92.06 |
   | TVIL（自建） | V | 88.68 | 84.70 | 62.43 | 87.09 | 88.21 | 90.43 | 91.16 |
   | Psynd | A | 100.00 | 100.00 | 79.87 | 97.60 | 97.60 | 97.60 | 97.60 |

   交叉验证：AuViRe 宣称的 "+8.9 AP@0.95 on LAV-DF" 与 **46.52 − 37.61 = 8.91** 完全吻合 → 两者数字互证【核实（两处独立来源）】。
7. **没解决的问题 / 局限**：
   - **"Universal" 的代价是"每个数据集/模态单独训一个模型"**：README 给的是 6 个独立 ckpt（V / V+A / TVIL / Psynd…），**没有任何"一个权重跨模态零样本"的实验**——"universal" 实际是"框架可配置"，不是"模型通用"。
   - **@0.95 随数据集剧烈波动**（LAV-DF V+A 37.61 vs Psynd 79.87 vs TVIL 62.43）：说明分数对**段长分布/域**极度敏感，论文未做归因。
   - **完全依赖预提特征（TSN/BYOL-A）**：特征层的伪造线索上限被锁死；AuViRe 正是以此为主要批评点（"general purpose feature extractors limiting representational capacity"）【核实，AuViRe §2】。
   - **TVIL 是自建数据集**（视频 inpainting），规模与公开性弱于 LAV-DF；Psynd 是**纯音频**，把"100/100"写进主表会**高估方法的通用性**。
   - **无跨库 zero-shot 主表**（只有同库），**无效率报告**，**无标签噪声/边界抖动的鲁棒性分析**——而它的 AP@0.95 恰恰最吃这两项。

---

### 卡片 6 ｜TALL（Thumbnail Layout）
**"TALL: Thumbnail Layout for Deepfake Video Detection"** — Xu et al.，**ICCV 2023**，arXiv:2307.07494【核实】

1. **任务定义**：**视频级二分类（real / fake）**，**不做时序定位**。目标是用"图像级骨干 + 布局技巧"逼近视频级检测的精度，同时把计算量压到图像级。
2. **输入表示**：**原始帧**。把一段**连续帧**（固定帧数）在每帧的**固定位置做 mask**，再各自 resize 成 sub-image，**重排成预定义网格（thumbnail layout）**，作为一张"大图"送进 **Swin Transformer**。骨干无关（model-agnostic），只需改几行代码。
3. **时序建模方式**：**隐式时空建模**——2D Swin 的**窗口自注意力**在布局上跨 sub-image 生效（窗口跨越多个 sub-image 时即建模"帧间不一致"）；**没有时间轴卷积/Transformer、没有边界回归、没有 proposal**。本质是把"时间"折进"空间布局"。
4. **监督信号与标签粒度**：**视频级 0/1 标签**（交叉熵）；**无帧级/段级监督**，因此**原理上无法输出"何时"**。
5. **评测口径**：**Acc + AUC**；**intra-dataset**（FF++ 各压缩档）+ **cross-dataset**（Celeb-DF、DFDC、DeeperForensics）；图像级骨干按**全视频预测平均**聚合成视频级（"we calculate video-level predictions for the image-based method and average the predictions across the entire video"）【核实】；同时报 **FLOPs / 参数量**。
6. **报告数字**：**FF++ → Celeb-DF 跨库 AUC 90.79%**（TALL-Swin，摘要原文）【核实】；另有 Table 1 的"骨干 × TALL"对照（含 FLOPs/params），其中 ResNet/EfficientNet/ViT/Swin 加 TALL 后 CDF、DFDC 上的 AUC 均提升——**具体每格数字【未核实】**（表格在抓取截断之外，README 无表）。
7. **没解决的问题 / 局限**：
   - **只回答"是不是"，不回答"哪一段"**：与本文的 segment-level 任务正交；把它当 TFL 基线只能靠"逐窗滑窗 + 阈值"，论文未做、也未讨论。
   - **固定帧数的 thumbnail** 意味着**长视频必须采样**，边界信息在采样时就被丢弃——这是"效率换精度"的隐性代价。
   - **无视音频**：对"只改音频"或"音视不同步"的伪造**结构性失效**，而 AV-Deepfake1M 里 audio_modified 占相当大的比例。
   - **mask 固定位置是强先验**：可被针对性攻击（攻击者只需在该位置保持真实），论文未做自适应攻击实验。
   - **跨库提升主要来自 FF++ 训练**：FF++ 的伪造类型（换脸/表情重演）与"局部片段篡改"分布不同，跨库 90.79 不能外推到 TFL 场景。

---

### 卡片 7 ｜【CFAD 替代】HBMNet
**"Localizing Audio-Visual Deepfakes via Hierarchical Boundary Modeling"** — Chen et al.，arXiv:2508.02000（2025，comment: "Work in progress"）【核实】
> **替代说明**：CFAD 无法在 arXiv/Crossref 落地（见 §0 卡片清单第 7 行）。本卡以 HBMNet 替代：它是 2025 年**音视频 TFL** 方法，摘要明确 "outperforms BA-TFD and UMMAFormer"，最适合承担"时序定位骨架族"的对照。若你后续能给出 CFAD 的确切出处（DOI/arXiv），本卡可原位替换，其余结构不变。

1. **任务定义**：**音视频 TFL**，聚焦"content-driven 局部篡改、伪造区仅数帧、其余与原始完全一致"的困难设定；输出**候选段（coarse proposals）+ 精修边界**。
2. **输入表示**：**音视频双流特征**（专用编码 + 融合），强调**帧级监督**以提升判别力（"frame-level supervision to boost discriminability"）。
3. **时序建模方式**：**两阶段（proposal + refinement）+ 多尺度 + 双向边界内容建模**：
   - **AV Feature Encoder**：帧级判别表征；
   - **Coarse Proposal Generator**：预测候选边界区间（BSN 式，论文明确 "Similar to BSN (Lin et al. 2018)" 定义 GT 时间区域）；
   - **Fine-grained Probabilities Generator**：用**双向 boundary-content 概率**精修 proposal；
   - 多尺度线索融合。
4. **监督信号与标签粒度**：**帧级标签 + 段级（proposal）标签**联合监督（摘要强调 frame-level supervision 提升 recall、融合提升 precision）。
5. **评测口径**：沿用 AV-TFL 的 **AP@IoU / AR@N** 家族（与 LAV-DF/AV-Deepfake1M 一致）；**具体阈值集合与是否用 soft-NMS【未核实】**。
6. **报告数字**：**摘要只给相对结论**（优于 BA-TFD 与 UMMAFormer；"shows improved potential scalability with more training data"），**绝对数字【未核实】**（表格在 100k 抓取截断之外）。
7. **没解决的问题 / 局限**：
   - **两阶段 proposal 的误差累积**：coarse proposal 漏掉的段，refinement 永远救不回来（正是本项目诊断出的"提案覆盖"瓶颈，见缺口 G3）。
   - **"Work in progress"**：无正式发表、无公开代码，数字不可复现。
   - **不报效率**：proposal 生成 + 双向精修的计算开销未与单阶段方法对照。
   - **未覆盖跨库与模态缺失**（video-only / audio-only）。
   - **作为 CFAD 的替代，它并不解决 CFAD 原本要问的"跨模态特征对齐/融合设计"**——这一点必须在论文里如实说明，不能拿 HBMNet 冒充 CFAD 的结论。

---

### 卡片 8 ｜ActionFormer
**"ActionFormer: Localizing Moments of Actions with Transformers"** — Zhang et al.，**ECCV 2022**，arXiv:2202.07925【核实】

1. **任务定义**：**TAL（Temporal Action Localization）**：输入**未裁剪视频的特征序列**，**单阶段（single shot）**地为**每个时刻**分类（动作类别）并回归**该时刻所属实例的起止边界**；**不用 proposal、不用预定义 anchor 窗口**。
2. **输入表示**：**预提特征**（I3D / TSP / VideoMAE 等）；短视频按 16 帧 clip、非重叠或固定 stride 提特征（README：ActivityNet 用 16 帧 clip @15fps、stride 16，即约 1.067 s 一个特征向量）。
3. **时序建模方式**：**多尺度特征金字塔 + 局部自注意力 Transformer 编码器 + 轻量解码器**。金字塔各级用**局部（windowed）自注意力**控制复杂度，解码器逐位置输出类别分数与边界回归量；推理时**整段前向**（因不使用位置编码）。
4. **监督信号与标签粒度**：**逐位置标签分配 + 稠密回归**：每个 GT 实例指派给"中心最近"的位置，回归目标为该实例的**归一化起止偏移**；损失 = **focal（分类）+ DIoU（回归）**；另用"分类分数 × 回归 IoU"的一致性加权。粒度 = **帧/点级 + 段边界**（标签是"实例级段"，不是逐帧二值）。
5. **评测口径**：**mAP at tIoU**（THUMOS14：tIoU 0.3–0.7，步长 0.1）；**average mAP**（ActivityNet 1.3：tIoU 0.5:0.05:0.95）；EPIC-Kitchens 用 average mAP；**提案数固定、无 soft-NMS**（单阶段直接输出）。
6. **报告数字**【核实】：
   - **THUMOS14：71.0% mAP @ tIoU=0.5**，比此前最好模型 **+14.1 个绝对点**（摘要）；
   - **ActivityNet 1.3：36.6% average mAP**（摘要）；
   - **EPIC-Kitchens 100：+13.5% average mAP**（摘要）；
   - 官方仓库复现（更新后的代码）THUMOS14 逐阈值：**82.13 / 77.80 / 70.95 / 59.40 / 43.87，Avg 66.83**（tIoU 0.3/0.4/0.5/0.6/0.7）【核实，README 表】；
   - Ego4D Moment Queries 2022 提交：**21.76% average mAP、42.54% Recall@1x @tIoU0.5**，排名第 2【核实，README】。
7. **没解决的问题 / 局限**：
   - **"动作先验"≠"伪造先验"**：它的标签是**语义类别**（有类别间差异可学），而 TFL 的"伪造"**无类别语义、段更短、真样本占比极高**；直接迁移会出现"学到动作边界而非篡改边界"（UMMAFormer 必须外挂 TFAA 重建分支来补偿，见卡片 5）。
   - **依赖离线预提特征**：不能端到端；特征分辨率（约 1 s/向量）**天然决定边界精度上限**，对"几帧级"的 deepfake 段太粗。
   - **显存/算力**：README 明确 "training ~4.5 GB, **inference >10 GB**"——长视频推理成本高【核实】。
   - **无校准、无"真/假"二分类头**：直接把"背景类"当 real 用会与"未篡改段"混淆（TFL 中 real 段占绝大多数，与 TAL 的长背景段同构但语义不同）。
   - **AP 的提案数上限固定**，不做 AR@N 曲线 → 与 TFL 论文的"AR@N"口径**不可直接对齐**（这也是 AV-Deepfake1M 表里 ActionFormer 行 AR@50 反而低于 BA-TFD 的原因之一）。

---

### 卡片 9 ｜DiMoDif
**"DiMoDif: Discourse Modality-information Differentiation for Audio-visual Deepfake Detection and Localization"** — Koutlis & Papadopoulos，arXiv:2411.10193（v2, 2025-04）【核实】

1. **任务定义**：**AV-DFD（视频级，且分模态判定）+ TFL（帧级 + 伪造区间）** 双任务。核心假设："真实样本中音视信息一致，深伪中不一致"。输入整段音视频，输出**逐帧跨模态不一致度 + 伪造区间**。
2. **输入表示**：**语音专用表征**——用预训练的 **VSR（视觉语音识别）** 与 **ASR（音频语音识别）** 模型提特征（DiFE 模块，"Discourse-related Feature Extraction"），**帧级对齐**（adaptive temporal alignment）；不使用原始像素/波形直接建模。
3. **时序建模方式**：**层级跨模态融合网络（hierarchical cross-modal fusion）+ 自适应时间对齐（adaptive temporal alignment）+ 学习式差异映射层（discrepancy mapping layer）**，显式建模"视觉嵌入 vs 音频嵌入"的细微差异；输出帧级分数 + 区间回归；损失含**帧级检测项 + 伪造区间定位项**的复合损失。
4. **监督信号与标签粒度**：**帧级标签 `{y^v_1..y^v_f, y^a_1..y^a_f}` + 视频级模态标签 `y ∈ {0,1}^2`**（视觉是否被改、音频是否被改）【核实，§Problem Formulation】。即：**视频级多标签 + 帧级二值**，边界以"区间"形式进入定位损失。
5. **评测口径**：DFD 用 **AUC**；TFL 用 **AP@IoU / AR@N**（沿用 LAV-DF、AV-Deepfake1M 的官方协议）。
6. **报告数字**【核实，摘要】：**AV-Deepfake1M 上 DFD +30.5 AUC**（相对 SOTA）；**TFL +47.88 AP@0.75**（AV-Deepfake1M）；**LAV-DF 上与 SOTA 持平**；FakeAVCeleb 表现优异。**绝对数值【未核实】**（各表落在抓取截断之外，官方仓库 README 未列表）。
7. **没解决的问题 / 局限**：
   - **"不一致=伪造"假设的盲区**：若音视频被**一致地**重录/重配音（同一个人、同内容），差异度不升 → 系统性漏检；论文只论证"真实样本中两者一致"，未给反例压力测试。
   - **强依赖 VSR/ASR 质量**：噪声、口音、遮挡、侧脸、非英语都会同时抬高"真实样本"的差异度 → **假阳性**；跨语言表现未评（AuViRe 的 in-the-wild 用了 39 种语言，DiMoDif 未做同类实验）。
   - **对齐模块的隐性假设**：音视在时间上"应当同步"；对**后期配音、画外音、音乐**样本，对齐本身失效。
   - **与 AuViRe 同组、同数据、同指标**：两者的差异主要被表述为 "encoding & comparison vs reconstruction"（AuViRe §2 原话）——**这条对比是"同一假设下的两种实现"，不是两条独立技术路线**，论文的 "fundamentally different" 表述偏强。
   - **无效率报告、无跨库 zero-shot 主表**、**无阈值校准**（与 AuViRe 同样的缺口）。

---

### 卡片 10 ｜AVFF
**"AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection"** — Oorloff et al.，**CVPR 2024**，arXiv:2406.02951【核实】

1. **任务定义**：**视频级 DFD（real/fake 二分类）**，**不做时序定位**。目标是学到"可迁移的音视对应关系"，而不是只拟合训练集里的伪造痕迹。
2. **输入表示**：**人脸视频帧 + 音频**；把音视频沿时间轴切成 **K 个时间切片（temporal slices）**，各自用**单模态 Transformer 编码器**（`E_a`、`E_v`）编码为 token 序列。
3. **时序建模方式**：**两阶段跨模态表征学习**：
   - 阶段一（自监督，只用真实视频）：**互补掩码**（对音/视 token 各掩 50%，掩码互补）+ **A2V 预测网络**（用可见音频 token 预测被掩的视觉 token）+ **对比损失 + 自编码损失**；
   - 阶段二：在真假视频上**监督微调**分类头。
   - 时序维度通过"时间切片 + 互补掩码"体现，**没有边界回归、没有 proposal**。
4. **监督信号与标签粒度**：阶段一 **自监督（无标签，仅真实视频）**；阶段二 **视频级 0/1 标签**。**完全没有帧级/段级监督**。
5. **评测口径**：**Acc + AUC**（视频级）；主战场 FakeAVCeleb，另做跨库（FF++/Celeb-DF/DFDC 类）泛化分析【部分未核实】。
6. **报告数字**【核实，摘要】：**FakeAVCeleb 上 98.6% Acc、99.1% AUC**，比当时 AV-SOTA **+14.9%（Acc）/ +9.9%（AUC）**。跨库逐项数字【未核实】。
7. **没解决的问题 / 局限**：
   - **只输出一个视频级分数**：对"视频里只有 2 秒被改"的场景，**梯度被整段稀释**——训练目标是"整段真/假"，与 segment-level 任务**目标函数不一致**；论文也未在"局部篡改"数据（LAV-DF / AV-Deepfake1M）上评测。
   - **自监督阶段吃"真实人脸视频"的规模与质量**：换成低资源语言/非人脸场景即退化。
   - **无声/纯视觉伪造**（无音轨或音轨与内容无关）时，A2V 预测无意义 → **结构性失效**。
   - **"对应关系"的可解释性未验证**：无法证明模型用的是唇音同步而非 TTS/生成器指纹；论文用 t-SNE/消融间接论证，未做"反事实"控制（例如把音频替换为同说话人真实音频）。
   - **作为"重建残差族"的一员，它其实没有显式的重建残差输出**（A2V 预测是训练目标，不是推理时的分数）——在 Gap 矩阵里被归入"重建族"是因为机制相近，**这一点在论文写作时要写清楚，否则会被审稿人抓**。

---

### 卡片 11 ｜MDS（Modality Dissonance Score）
**"Not made for each other – Audio-Visual Dissonance-based Deepfake Detection and Localization"** — Chugh et al.，**ACM MM 2020**，arXiv:2005.14405【核实】
> **重要澄清**：AV-DFD 文献里的 "MDS" 就是本文提出的 **Modality Dissonance Score**（AVFF 与 BA-TFD 的 Related Work 均如此引用："proposed the Modality Dissonance Score (MDS) network, where a contrastive loss is computed on single modality embeddings"）【核实】。它不是 MDS-Net，也不是 Multi-Dimensional Scaling。

1. **任务定义**：**AV-DFD（视频级）+ 最早的 TFL 尝试之一**。把视频切成 **1 秒音视频 chunk**，逐 chunk 计算"模态不协调度" MDS，聚合成视频级分数判真/假；**通过阈值化 chunk 分数得到伪造段**（作者自述 "to our knowledge, our method is the first to achieve temporal forgery localization"）【核实】。
2. **输入表示**：**1 秒音视频片段** → 视觉子网（人脸帧）与音频子网（频谱）分别提特征；**不用预训练大模型**（2020 年的 CNN 前端）。
3. **时序建模方式**：**片段级 + 对比学习**：对单模态嵌入计算 **contrastive loss**（拉近真实音视对、推远不匹配对），MDS = chunk 分数的聚合（平均）；**没有时序卷积/RNN/Transformer，没有边界回归**——"时序"完全由"1 秒切片 + 聚合"实现。
4. **监督信号与标签粒度**：**弱监督**：只用**视频级真/假标签**（对比损失 + 单模态交叉熵），**没有帧级/段级标注**；段级输出是"阈值化后的副产品"，**不是被监督出来的**。
5. **评测口径**：视频级 **AUC / Acc**（DFDC、DF-TIMIT、FaceForensics 等）；判定阈值 `τ` 取"训练集真实视频 MDS 均值与伪造视频 MDS 均值的中点"（`1{MDS_i < τ}`）【核实】；定位评测的 IoU 口径【未核实】。
6. **报告数字**：摘要称在 DFDC / DF-TIMIT / FaceForensics 上 **"outperforms the state-of-the-art by up to 7%"**（原文措辞，未指明是 AUC 还是 Acc）【核实为原文措辞，具体数值未核实】；其余逐项数字【未核实】。
7. **没解决的问题 / 局限**：
   - **1 秒粒度 = 边界精度天花板 ±1 s**：在高 IoU 阈值（0.9/0.95）上**结构性不可行**，这也是此后所有 chunk-level 方法在 TFL 上被 proposal/回归方法碾压的原因。
   - **阈值 `τ` 由训练集统计量决定**：**不可校准、不可迁移**（换数据集即失效），而论文把"定位"建立在同一个 `τ` 上。
   - **只针对"唇音不同步"类伪造**：对纯视觉换脸（音频真实）、纯音频替换（画面真实）**判据失效**——恰恰是后来 AV-Deepfake1M 里占多数的两类。
   - **无 proposal、无排序、无 NMS**：多段篡改时无法给出"第 k 个段"，AR@N 意义下不可比。
   - **"定位"的评测强度弱**：没有公开的段级 AP/AR 协议（2020 年尚无 LAV-DF），定位结论主要靠可视化。

---

### 卡片 12 ｜AVoiD-DF
**"AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfake"** — Yang, Zhou, Chen, Guo, Ba, Xia, Cao, Ren，**IEEE TIFS, vol. 18, pp. 2015–2029, 2023**，DOI 10.1109/TIFS.2023.3262148【核实（Crossref + 挑战赛论文参考文献列表双重确认）】

> **诚实声明**：本卡是本文件中**证据最弱**的一张。IEEE Xplore 全文不可达（跨域重定向被拦），arXiv 无版本（`all:"AVoiD"` 检索无命中），Semantic Scholar 限流。因此**只有书目元数据是核实的**，技术细节一律【未核实】。若要把它写进正式 Related Work，**必须补读 IEEE 原文**，否则建议改用卡片 11（MDS）承担"不一致性族"的论证。

1. **任务定义**：**音视频联合学习的深伪检测（AV-DFD）**，面向**视频级真/假判定**；是否含段级定位头【未核实】（从题名与 TIFS 2023 的定位看，应属检测分类族，但**不做断言**）。
2. **输入表示**：【未核实】（推测为人脸序列 + 音频频谱的联合表征，但**无来源支撑，不作为结论**）。
3. **时序建模方式**：【未核实】。
4. **监督信号与标签粒度**：【未核实】（视频级标签可基本确定，帧/段级未知）。
5. **评测口径**：【未核实】（大概率 AUC/Acc 视频级；无证据表明使用 AP@IoU/AR@N）。
6. **报告数字**：【未核实】。
7. **没解决的问题 / 局限**：
   - **可复现性缺口**：无 arXiv 版本、无公开代码链接（本轮检索未见），第三方难以复核；
   - **在本文语境下的结构性局限（可确证的部分）**：作为 2023 年的"检测分类族"工作，它**没有段级监督/段级输出**，因此**无法回答"何时"**；在 1M-Deepfakes 挑战赛的 baseline 综述里它被列为**检测侧**参照，而非定位侧【核实，挑战赛论文参考文献与 baseline 讨论】；
   - **其余局限【未核实】**，本文件不做推测。

---

## 2. Gap 矩阵

**读法**：每格写"**谁部分填了这个格子**（证据）+ 填到什么程度"；写"**空**"表示**该族内没有任何一篇论文处理过这个维度**。族内成员：重建残差族 = {AuViRe, DiMoDif, AVFF}；时序定位骨架族 = {UMMAFormer, ActionFormer, TALL}；检测分类族 = {BA-TFD/BA-TFD+, LAV-DF, AV-Deepfake1M}；不一致性族 = {AVoiD-DF, MDS, CFAD→HBMNet}。

| 维度 | 重建残差族（AuViRe / DiMoDif / AVFF） | 时序定位骨架族（UMMAFormer / ActionFormer / TALL） | 检测分类族（BA-TFD / LAV-DF / AV-Deepfake1M） | 不一致性族（AVoiD-DF / MDS / CFAD→HBMNet） |
|---|---|---|---|---|
| **边界精度** | **AuViRe**：回归头 + DIoU 逐帧回归起止（唯一把边界当"回归量"的重建族方法），LAV-DF AP@0.95 **46.52**；但 @0.9→@0.95 从 72.10 掉到 46.52，说明"亚帧级"仍未达成。DiMoDif 只回归"区间"且在 LAV-DF 与 SOTA 持平（未超）。**AVFF 空**（无边界输出） | **UMMAFormer**：ActionFormer 式逐位置回归 + PCA-FPN，LAV-DF V+A AP@0.95 **37.61**；**ActionFormer**：DIoU 回归，是这套"逐位置回归"的源头；**TALL 空**（无边界概念）。共同短板：特征分辨率约 1 s（TSN/I3D），**边界精度被特征粒度锁死** | **BA-TFD/BA-TFD+**：边界图（MSE）路线，AP@0.95 仅 **0.24 → 4.44**，是本矩阵中"边界精度"最弱的一族；**LAV-DF/AV-Deepfake1M 本身不提供方法**，只提供标注（且 1M 表里全族 AP@0.95 ≤ 1.58）→ 该格实质是"**数据给了、方法没解**" | **HBMNet**（CFAD 替代）：显式"coarse proposal + 细粒度边界精修" + 双向 boundary-content，是族内唯一把边界当一等公民的；**MDS 空**（1 秒 chunk，±1 s 天花板）；**AVoiD-DF 空**（无定位头） |
| **排序 / 打分校准** | **AuViRe**：soft-NMS 后取 `max ρ_j` 作视频级分数，**但固定 0.5 阈值下 Acc 仅 73.54 vs AP 99.98**（本项目实测）→ 分数尺度无语义；DiMoDif/AVFF 均未做校准分析（AVFF 只报 Acc/AUC，AUC 对单调变换免疫，**恰好掩盖了校准问题**） | **ActionFormer/UMMAFormer**：用"分类分 × 回归 IoU"做一致性加权（**隐式校准**，是族内唯一有"分数质量"意识的设计），但**没有可靠性图/ECE/温度标定**；**TALL 空**（单分数二分类） | **BA-TFD/LAV-DF**：无排序设计（边界图阈值化）；**AV-Deepfake1M**：官方综合分 `S` 把 AP 与 AR 直接相加（`ΣAP/8 + ΣAR/10`），**这一步在指标层面把"排序"与"覆盖"混为一谈**，反而掩盖了校准问题 | **MDS**：`τ` = 训练集真/假均值中点（**手工校准，不可迁移**）；**HBMNet**：proposal 分数 + 精修概率，无校准分析；**AVoiD-DF 空** |
| **提案覆盖（recall）** | **AuViRe**：AR@100 **94.93** vs AP@0.95 **46.52**（LAV-DF）→ 召回不是瓶颈，**但本项目诊断显示 10.7% 的段在 soft-NMS 后"不可达"**（oracle 上限 89.3%，本项目实测）→ 覆盖仍有硬缺口；DiMoDif/AVFF **空**（无 proposal 概念） | **UMMAFormer**：AR@100 92.48（LAV-DF V+A），但 AR@10 已 92.10 → **曲线饱和，说明"多给提案"没收益**；**ActionFormer**：AR 定义与 TFL 不同（无 AR@N 曲线，只有 mAP）；**TALL 空** | **BA-TFD**：AR@100 **67.03**（LAV-DF）明显低于 UMMAFormer/AuViRe → 边界图范式**召回本身就不够**；**AV-Deepfake1M**：AR@50/20/10/5 全表都比 AP@0.95 高一个数量级（如 UMMAFormer AR@50 44.07 vs AP@0.95 1.58）→ **数据侧证明"召回易、精确难"** | **HBMNet**：coarse proposal generator 的召回未单独报告（【未核实】）；**MDS 空**；**AVoiD-DF 空** |
| **跨数据集泛化** | **AuViRe**：**唯一给出双向跨库数字的工作**——LAV-DF→1M AP@0.5 **16.05**（同库 98.81）、1M→LAV-DF **53.76**（同库 98.91）；Codabench 上 LAV-DF 训练仅 **0.147 AP@0.5 / 0.657 AUC**（本项目实测）→ **断崖被量化，但未被解决**。DiMoDif：只在三库各自报，无跨库主表；AVFF：主打"可迁移表征"，但评测是视频级分类、**不在 TFL 上验证** | **UMMAFormer/ActionFormer**：**无跨库 TFL 实验**（ActionFormer 的跨库是 TAL，非同任务）；**TALL**：有跨库（FF++→CDF 90.79 AUC），但**是分类任务**，不能外推到 TFL | **LAV-DF/AV-Deepfake1M**：两库本身构成"跨库对"，但**数据集论文不研究泛化**；BA-TFD+ 只在各自库内报 | **MDS**：`τ` 不可迁移 → 跨库直接失效；**HBMNet**：无跨库报告；**AVoiD-DF**：未核实 |
| **效率** | **AuViRe**：需**离线预提 AV-HuBERT 特征**（冻结、768 维），单模型仅 1D 卷积（轻），但**特征提取是重前端**；参数量/FLOPs/吞吐【未核实】（正文 §5.6 被截断）；DiMoDif 双语音模型（VSR+ASR）→ 更重；**AVFF**：两阶段（自监督预训练 + 微调）+ 双 Transformer → 训练成本最高 | **ActionFormer**：README 明确 "训练 ~4.5 GB、**推理 >10 GB**"（长视频不可行）；**UMMAFormer**：Transformer + FPN + 跨注意力，成本高于 1D 卷积；**TALL**：**唯一系统性报 FLOPs/params 的工作**（"简单 + 高效"是其核心卖点）→ 效率维度**只有 TALL 真正填了** | **BA-TFD**：3D CNN（时空卷积）→ 显存/算力最重；BA-TFD+ 换 MViT 更重；**LAV-DF/AV-Deepfake1M**：数据生成约 3000 GPU 小时（1M），**推理效率未涉及** | **MDS**：1 秒 chunk + 小 CNN，**最轻**（但精度最差）；**HBMNet**：两阶段 proposal + 精修 → 中重；**AVoiD-DF**【未核实】 |
| **标签粒度 / 噪声** | **AuViRe**：**混合粒度**（帧级二值 + 秒级边界），重建正则以"整段为真"为条件；**本项目实测：把整数帧标签换成连续标签，AP@0.95 从 50.60 → 83.15（+32.6 pp）**，而论文**完全没有讨论标签定义**；DiMoDif：帧级 + 视频级模态标签（粒度最细）；AVFF：**只有视频级标签**（粒度最粗，与 segment 任务错配） | **UMMAFormer**：逐点二值（段内=1、段外=0）+ 边界回归，**对边界抖动无鲁棒项**；**ActionFormer**：实例级段标签 + 最近位置分配（**对"极短段"分配不稳定**）；**TALL**：视频级，**空**（不涉及） | **LAV-DF**：**帧索引整数区间**（噪声源）；**AV-Deepfake1M**：**秒级 `fake_segments`**（比 LAV-DF 粗，但更接近人工标注语义）；两者都**没有标注置信度/多人一致性**；**AV-Deepfake1M 的人类 AP@0.5 仅 1.92** → **标签本身不可辨，是"噪声"的直接证据** | **MDS**：只有视频级标签，段是阈值化副产物（**弱监督**）；**HBMNet**：帧级 + proposal 级（口径较全）；**AVoiD-DF 空** |

**矩阵一句话总结**：
- **边界精度**：只有 AuViRe / UMMAFormer / HBMNet 在真做，其余族要么没有边界输出（TALL/AVFF/MDS），要么做不出来（BA-TFD 0.24）；
- **排序/校准**：**四个族全是半空**——最好的也只是"分类分 × IoU"的隐式加权（ActionFormer/UMMAFormer），**没有一篇做过分位数校准/可靠性分析**；
- **提案覆盖**：AuViRe 的 AR@100 94.93 看似填满，但 **AR@5=92.16 说明曲线早饱和**，真正的瓶颈是"top-1 命中"与"不可达段"（10.7%）；
- **跨数据集泛化**：**只有 AuViRe 给了双向跨库 TFL 数字，且数字本身就是"断崖"**；
- **效率**：**只有 TALL 系统报 FLOPs**，其余全是"未报告"；
- **标签粒度/噪声**：**数据侧有证据（人类 AP@0.5=1.92、整数帧 vs 连续标签 +32.6 pp），方法侧零处理**。

---

## 3. 候选研究缺口（5 条，含"这条缺口是真的"的证据）

### G1 ｜标签粒度轴与评测轴错配：**没有人定义"什么是段边界"**（最强的一条）
**缺口陈述**：所有 TFL 工作都默认"GT 段 = 生成流水线的拼接点"，并用**帧索引整数**标注；但没有一篇论文讨论"整数帧 vs 连续秒"对 **高 IoU 阈值指标（@0.9/@0.95）** 的影响，也没有任何方法对**边界抖动/标注噪声**建模。

**证据（缺口真实）**：
1. **本项目 A/B（同 seed、同数据、同模型，仅换标签定义）**：整数帧标签 AP@0.95 = **50.60**，连续标签 = **83.15**（+32.6 pp）；@0.5/@0.75 几乎不动（98.64→98.71 / 95.73→95.84）【本项目实测】。→ **@0.95 上的大部分"方法差异"其实是"标签口径差异"**。
2. **oracle 可达率**：soft-NMS 后的提案集合里，**89.3% 的 GT 段存在一个 IoU≥0.95 的提案**，但模型只选中 71.9%（整数标签）/ 78.7%（连续标签）→ **10.7% 的段在任何解码策略下都拿不到**，其余是**排序损失**。
3. **人类基线**：AV-Deepfake1M 用户研究 AP@0.5 = **1.92**、AR@1 = 2.54【核实】→ 人给出的边界与 GT 几乎不重合，**说明标签边界不是感知可辨的事件**。
4. **文献侧**：AuViRe 的 GT 定义是 `p∈{0,1}^t` + `b∈[0,Δ]^(t×2)`（**混合粒度**，且 `b` 只在伪造帧有效）【核实】；LAV-DF 用帧索引区间 + 边界图 MSE【核实】；AV-Deepfake1M 用秒级 `fake_segments`【核实】——**三套口径互不兼容，却共用同一套 AP@IoU 指标**。

**可做的贡献**：把"标签轴"显式化为方法的一部分（连续时间回归 / 分布化边界 / 软标签 + 不确定性），并给出"标签-指标一致性"分析协议。**风险**：可能被审稿人视为"评测 trick"——必须配"方法侧"改动（如边界分布头 + 期望 IoU 损失）。

---

### G2 ｜高分位（@0.9/@0.95）的**排序与打分校准**是四族共同的空白
**缺口陈述**：所有方法在 AP@0.5 都 ≥ 96，但 @0.95 分布在 **0.24（BA-TFD）~ 46.52（AuViRe）** 之间；没有人回答"**为什么 top-1 段选错**"，也没有人做**置信度校准**（温度标定 / 可靠性图 / 分位数回归）。

**证据**：
1. **同库高分位塌陷**：LAV-DF 上 AuViRe 98.91→96.03→72.10→**46.52**（@.5/.75/.9/.95）；UMMAFormer 98.83→95.54→**37.61**；BA-TFD+ 96.30→84.96→**4.44**【核实，官方 README】。
2. **召回不缺**：AuViRe AR@100 **94.93** vs AP@0.95 **46.52**；AR@5 已 **92.16** → **"多给提案"没有边际收益**，问题在**排序**。
3. **校准缺失的直接证据**：DFD 侧 AP **99.98** 但固定 0.5 阈值的 Acc 仅 **73.54**（本项目复现，`src/eval.py:161` 的阈值写死）【本项目实测】→ **分数绝对值无意义**，任何"报警阈值"都无法跨库迁移。
4. **指标侧反而掩盖问题**：AV-Deepfake1M 官方综合分 `S` 把 AP 与 AR 相加（`ΣAP/8+ΣAR/10`）【核实】，**AR 项把"排序失败"摊平了**；AVFF 只报 AUC（**对单调变换免疫**）【核实】。
5. **文献侧**：四族中**没有一篇**报告 ECE / 可靠性图 / 分数分布；唯一沾边的是 ActionFormer 的"分类分 × IoU"一致性加权（隐式，且未评估）。

**可做的贡献**：把"定位"重新表述为**段级排序/校准问题**（learning-to-rank + 分位数/共形校准），指标上加 **AP@0.95 + ECE + top-1 命中率**。**这条最"干净"**：不依赖标签定义，且能同时解释"AR 高 AP 低"。

---

### G3 ｜提案覆盖的**硬缺口**：不可达段（unreachable segments）无人处理
**缺口陈述**：大家都在优化"打分"，但**soft-NMS 之后 GT 段根本没有对应提案**的情形（10.7%）**没有任何论文讨论**；两阶段方法（HBMNet）有 proposal 但**不报 proposal recall**，单阶段方法（UMMAFormer/AuViRe）把 proposal 生成藏在多尺度头里。

**证据**：
1. **本项目诊断**：oracle 可达率 **89.3%**（连续标签）/ **71.9%**（整数标签）→ **不可达段占 10.7%~28.1%**，这部分**无论怎么调排序都救不回来**（`AUVIRE_BOUNDARY_DIAG` 只读诊断）【本项目实测】。
2. **AR 曲线饱和**：UMMAFormer AR@10 92.10 vs AR@100 92.48（+0.38）；AuViRe AR@5 92.16 vs AR@100 94.93 → **继续加提案几乎无效**，说明"缺的不是数量，是覆盖位置"。
3. **BA-TFD 的召回本身就低**：LAV-DF AR@100 仅 **67.03**（vs UMMAFormer 92.48）→ **边界图范式的召回上限**被数据集论文自己暴露。
4. **HBMNet 明确用 coarse proposal + 精修**，但**摘要与（可见的）方法节均未报告 proposal recall**，无法判断其覆盖是否解决该问题【未核实】。
5. **段长分布**：局部篡改"通常只有几帧"（HBMNet 摘要原话）【核实】→ 短段在金字塔高层**直接消失**（下采样 2^λ），这是**结构性不可达**，而非训练不足。

**可做的贡献**：显式的"覆盖感知"训练目标（如 set prediction / 稠密 proposal + recall-oriented 采样），并**把 proposal recall 作为一等指标**报告。**风险**：与 DETR 式 set prediction 的收益/成本比需先验证（本项目已记录"本地最多 2 段/视频、相邻间隔中位 77 帧 → 收益/成本比低"）。

---

### G4 ｜跨数据集泛化：**只有一份"断崖"数据，没有任何"解法"**
**缺口陈述**：TFL 领域的跨库数字**几乎不存在**；唯一系统给出双向跨库的是 AuViRe，而它的结论是"崩"。没有人研究**域偏移的归因**（是特征域、段长分布、还是 TTS 指纹）。

**证据**：
1. **AuViRe 双向跨库**【本项目实测】：LAV-DF→1M AP@0.5 **16.05**（同库 98.81，**−82.8**）；1M→LAV-DF **53.76**（同库 98.91，**−45.2**）；**Codabench 真 test 上 LAV-DF 训练只有 0.147 AP@0.5、0.657 AUC**（几乎随机）。
2. **其他族**：UMMAFormer/ActionFormer/HBMNet/DiMoDif **均无跨库 TFL 主表**；BA-TFD+ 只在各自库内；TALL 的跨库是**分类任务**（90.79 AUC），不可外推。
3. **数据集侧的域差是可量化的**：LAV-DF（78,703/31,501/26,100）与 AV-Deepfake1M（746,180/57,340/343,240，test 无标签）**量级差 10 倍**，且 1M 用 VITS/Whisper 生成、LAV-DF 用更早的 TTS → **生成器指纹是混杂变量**，无一篇做控制实验（同生成器、不同数据）。
4. **反向迁移（1M→LAV-DF）反而好于正向**：53.76 vs 16.05【本项目实测】→ 说明**"大库→小库"可行、"小库→大库"不可行**，这本身是一个未被解释的现象（可能是段长/分辨率/语言分布）。

**可做的贡献**：跨库归因框架（控制生成器、段长、语言、分辨率）+ 域泛化目标（不变风险最小化 / 特征解耦）。**风险**：需要重训多组、算力成本高（本项目单 epoch 78,703 条在 4090 上 5–10 min，但 val+test+metrics 很慢）。

---

### G5 ｜效率与"模态缺失"：**部署侧的两个空白**
**缺口陈述**：TFL 方法普遍假设"**双流齐全 + 可离线提特征 + 视频不超过 512 帧**"；没有人给**流式/在线**方案，也没有人处理**audio-only / video-only / 无语音**的退化情形。

**证据**：
1. **AuViRe**：必须**离线预提 AV-HuBERT 特征**（仓库 README 明确要求先跑 `inference.py` 提特征）【核实】；序列上限 **512 帧**、超出即切窗，**长视频拼接协议缺失**（in-the-wild 中位 1 分钟、最长 11.3 分钟）【核实】；参数量/FLOPs【未核实】。
2. **ActionFormer**：README 明确 "**推理 >10 GB**" 显存【核实】；**TALL** 是唯一系统报 FLOPs 的（"简单高效"即为卖点）【核实】。
3. **模态缺失**：AV-Deepfake1M 的 `modify_type` 含 audio_modified / visual_modified 单独篡改【核实】，但 AuViRe/DiMoDif 均**只在双流同现时评测**；UMMAFormer 声称 universal（V/A/V+A），但**每种模态单独训一个模型**（README 6 个 ckpt）【核实】——**"通用"实为"可配置"**。
4. **场景假设**：AuViRe 的 real-world 实现要求 "talking human presence"【核实】；DiMoDif/MDS 均以"语音在场"为前提 → **非语音段（音乐、画外音、无脸）零覆盖**。

**可做的贡献**：模态自适应（missing-modality training）+ 流式/滑窗推理协议 + 统一效率报告（FPS/显存/特征提取占比）。**风险**：工程量大、创新度偏低，适合作为"系统贡献"而非"方法贡献"。

---

## 4. 横向对照表（写 Related Work 时可直接改用）

| 论文 | 年/venue | 任务 | 输入 | 时序机制 | 标签粒度 | 主指标 | 最强数字 | 最弱环节 |
|---|---|---|---|---|---|---|---|---|
| **AuViRe** | WACV 2026 | TFL(+DFD) | AV-HuBERT 冻结特征（分模态） | 1D conv 重建 + 残差 + FPN + 回归/分类双头 + SoftNMS | 帧级二值 + 秒级边界 | AP@IoU / AR@N | LAV-DF AP@0.95 **46.52** | 跨库 16.05 / 长视频与阈值校准缺失 |
| **LAV-DF (BA-TFD)** | DICTA 2022 | 数据集 + TFL/DFD | 原始帧 + log-mel | 3DCNN + 边界图 + 多模态融合 | 帧级 + 边界图（软） | AP@IoU / AR@N | AP@0.5 **79.15** | AP@0.95 **0.24** |
| **AV-Deepfake1M** | ACM MM 2024 | 数据集 + benchmark | 原始音视频 | 各基线自带（AV-LSTM 未核实） | 秒级段 + 视频级 | AP@IoU / AR@N / AUC / 综合分 S | 人类 AP@0.5 1.92（难度证据） | 全表 AP@0.95 ≤ 1.58；test 标签封闭 |
| **BA-TFD/BA-TFD+** | CVIU | TFL + DFD | 原始帧 + 音频 | 3DCNN（→MViT）+ 边界图 + 4 项损失 | 帧级 + 边界图 | AP@IoU / AR@N | BA-TFD+ AP@0.5 **96.30** | AP@0.95 **4.44** |
| **UMMAFormer** | ACM MM 2023 | 通用 TFL | TSN + BYOL-A 预提特征 | Transformer + TFAA 重建 + PCA-FPN + 逐位置头 | 逐点二值 + 边界回归 | AP@IoU / AR@N | LAV-DF V+A AP@0.95 **37.61** | 每模态单独训；@0.95 域敏感 |
| **TALL** | ICCV 2023 | 视频级分类 | 帧 → thumbnail 布局 | Swin 窗口注意力（隐式时序） | 视频级 | Acc / AUC（+FLOPs） | FF++→CDF AUC **90.79** | 无时间定位；无视音频 |
| **HBMNet**（CFAD 替代） | arXiv 2025 | AV-TFL | 音视频双流特征 | coarse proposal + 双向边界精修 + 多尺度 | 帧级 + proposal 级 | AP@IoU / AR@N（未核实细节） | 优于 BA-TFD/UMMAFormer（无绝对数） | 两阶段误差累积；无跨库/效率 |
| **ActionFormer** | ECCV 2022 | TAL | I3D/TSP/VideoMAE 特征 | 多尺度局部自注意力 + 逐位置 cls/reg | 实例段 + 最近位置分配 | mAP@tIoU | THUMOS14 **71.0 mAP@0.5** | 推理 >10 GB；特征粒度约 1 s |
| **DiMoDif** | arXiv 2024/25 | AV-DFD + TFL | VSR + ASR 语音特征 | 层级跨模态融合 + 自适应对齐 + 差异映射 | 帧级 + 视频级模态标签 | AUC / AP@IoU | 1M 上 **+30.5 AUC / +47.88 AP@0.75** | 依赖语音；绝对数未核实 |
| **AVFF** | CVPR 2024 | 视频级 DFD | 人脸帧 + 音频切片 | 互补掩码 + A2V 预测 + 对比/自编码 → 微调 | 自监督 + 视频级 | Acc / AUC | FakeAVCeleb **99.1 AUC** | 无定位；局部篡改未评测 |
| **MDS** | ACM MM 2020 | AV-DFD + 早期定位 | 1 秒 AV chunk | 片段级对比（MDS）+ 聚合 | 视频级（弱监督） | AUC / Acc | 较 SOTA "up to 7%"（未核实数值） | 1 秒粒度 → @0.95 结构性不可能 |
| **AVoiD-DF** | TIFS 2023 | AV-DFD | 未核实 | 未核实 | 未核实 | 未核实 | 未核实 | 无定位（据题名与引用位置推断） |

---

## 5. 参考文献（可核实链接）

1. Koutlis & Papadopoulos. *AuViRe: Audio-visual Speech Representation Reconstruction for Deepfake Temporal Localization.* WACV 2026. https://arxiv.org/abs/2511.18993 ｜代码 https://github.com/mever-team/auvire
2. Cai et al. *Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization.* DICTA 2022. https://arxiv.org/abs/2204.06228 ｜数据/代码 https://github.com/ControlNet/LAV-DF
3. Cai et al. *AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset.* ACM MM 2024. https://arxiv.org/abs/2311.15308 ｜https://github.com/ControlNet/AV-Deepfake1M ｜挑战赛 https://deepfakes1m.github.io/2024/evaluation
4. Cai et al. *Glitch in the Matrix: A Large Scale Benchmark for Content Driven Audio-Visual Forgery Detection and Localization.*（CVIU，审稿版）https://arxiv.org/abs/2305.01979
5. Zhang et al. *UMMAFormer: A Universal Multimodal-adaptive Transformer Framework for Temporal Forgery Localization.* ACM MM 2023. https://arxiv.org/abs/2308.14395 ｜https://github.com/ymhzyj/UMMAFormer
6. Xu et al. *TALL: Thumbnail Layout for Deepfake Video Detection.* ICCV 2023. https://arxiv.org/abs/2307.07494 ｜https://github.com/rainy-xu/TALL4Deepfake
7. Chen et al. *Localizing Audio-Visual Deepfakes via Hierarchical Boundary Modeling (HBMNet).* arXiv 2025. https://arxiv.org/abs/2508.02000 ｜（CFAD 的替代，见 §0）
8. Zhang et al. *ActionFormer: Localizing Moments of Actions with Transformers.* ECCV 2022. https://arxiv.org/abs/2202.07925 ｜https://github.com/happyharrycn/actionformer_release
9. Koutlis & Papadopoulos. *DiMoDif: Discourse Modality-information Differentiation for Audio-visual Deepfake Detection and Localization.* arXiv 2024. https://arxiv.org/abs/2411.10193 ｜https://github.com/mever-team/dimodif
10. Oorloff et al. *AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection.* CVPR 2024. https://arxiv.org/abs/2406.02951
11. Chugh et al. *Not made for each other – Audio-Visual Dissonance-based Deepfake Detection and Localization (MDS).* ACM MM 2020. https://arxiv.org/abs/2005.14405
12. Yang, Zhou, Chen, Guo, Ba, Xia, Cao, Ren. *AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfake.* IEEE TIFS 18: 2015–2029, 2023. DOI: 10.1109/TIFS.2023.3262148
13. （补充，仅元数据核实）Anshul et al. *Intra-Modal and Cross-Modal Synchronization for Audio-Visual Deepfake Detection and Temporal Localization.* ICCV 2025. DOI: 10.1109/ICCV51701.2025.01283
14. （补充）Cai et al. *1M-Deepfakes Detection Challenge.* arXiv:2409.06991（官方综合分 S 的定义出处）
15. （补充）Yin et al. *Context-aware TFL: A Universal Context-aware Contrastive Learning Framework for Temporal Forgery Localization.* arXiv:2506.08493（若日后要替换 CFAD，此为可用的"跨模态 TFL"候补）
16. （补充）*AV-Deepfake1M++.* arXiv:2507.20579（1M 的扰动增强版；本轮未用于结论）

---

## 6. 未核实清单（集中声明，写论文前必须补齐）

| 项 | 状态 | 原因 | 补齐方式 |
|---|---|---|---|
| "LAV-DF: Lips Are All You Need for Deepfake Detection" 一文 | **不存在/无法定位** | arXiv 全文检索 0 命中；Crossref 无匹配 | 若你有 PDF/DOI，请直接给我 |
| "BA-TFD: A Bi-modal Audio-Visual Transformer" 这一标题 | **与原文不符** | 官方全称为 Boundary Aware Temporal Forgery Detection | 已在卡片 4 更正 |
| CFAD | **无法定位** | arXiv/Crossref 均无 | 已用 HBMNet 替代并声明 |
| AV-LSTM 的结构与全部数字 | 未核实 | 论文 benchmark 表在 100k 抓取截断外；README 未列 | 需读 ACM DL 版 PDF 表 6/7 |
| AuViRe 计算效率（参数量/FLOPs/吞吐） | 未核实 | 正文 §5.6 被截断 | 读 PDF |
| AuViRe 正文 Limitations 段落 | 未核实 | 同上 | 读 PDF；卡片 1 第 7 条的 6 点是**我的推断**，非作者原话 |
| DiMoDif / AVFF / HBMNet / UMMAFormer(1M 外) / MDS / AVoiD-DF 的绝对 AP/AUC | 未核实 | 表格被截断或无公开全文 | 读各自 PDF |
| TALL Table 1 每格 AUC/FLOPs | 未核实 | 表格被截断 | 读 PDF |
| AVoiD-DF 全部技术细节 | 未核实 | IEEE 不可达、无 arXiv | 读 IEEE 原文 |
| AV-Deepfake1M 的 test 集规模 343,240 | 转述自 AuViRe | 未见 1M 原文对应表 | 交叉核对两文 |

---

### 附：一分钟版结论（给 parent agent 用）

1. **最强、最可写的一条 gap 是"标签粒度轴 ≠ 评测轴"**：整数帧 → 连续标签在 AP@0.95 上是 **+32.6 pp**（本项目实测 50.60→83.15），而**全领域没有一篇论文讨论标签定义**；人类 AP@0.5 只有 **1.92**（AV-Deepfake1M 用户研究，已核实）是最硬的旁证。
2. **第二条是"高分位排序/校准"**：AR@5 已 92.16 而 AP@0.95 只有 46.52（AuViRe, LAV-DF）→ 召回不缺、排序缺；DFD 侧 AP 99.98 但固定阈值 Acc 仅 73.54 → **校准完全空白**。
3. **第三条是"不可达段"**：oracle 可达率 89.3%，**10.7% 的段在解码后根本不存在**，无人处理。
4. **跨库只有一个"断崖"（16.05 vs 98.81），没有解法**；**效率只有 TALL 认真报过**。
5. 写作建议：Related Work 按"重建残差族 / 时序定位骨架族 / 检测分类族 / 不一致性族"四族组织（与本文件矩阵同构），并在每族末尾各留一句"该族未解决 X"，直接接到 G1–G3。
