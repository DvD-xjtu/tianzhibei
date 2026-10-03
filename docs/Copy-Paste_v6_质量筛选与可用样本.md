# Copy-Paste v6：质量筛选与小批量交付

日期：2026-09-24。此轮只核查**粘贴后图像质量**，不处理 15 类映射，也不启动训练。

## 结论

另以随机种子 `20260925` 在 MAR20 **train** 场景生成 40 张，逐张看原图/合成图对照后，交付 **20 张通过、20 张拒绝（50%）** 的独立小批量。最终样本放在：

`/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v6_mar20_40_curated_v3/`

其中 `train/images/` 和 `train/labels/` 是完整的 MAR20 YOLO-OBB 图像与标签；每张保留背景原有框，追加一条合成飞机框。`comparison/train/images/` 是原图/合成图对照；`instance_masks/`、`manifest.jsonl`、`qa/decisions.jsonl`、`qa/summary.json` 和 `qa/contact_sheets/` 供追溯。原始数据与未筛选 40 张均未覆盖。

## 怎么筛

1. 自动门：对已抠前景，计算 mask 内与原图 mask 外邻域颜色过近的像素比例；比例超过 **0.40** 就拒绝，防止把大量原背景一起贴入。本轮自动门拒绝 2 张。原有生成器仍先检查落点与原框碰撞、尺度、纹理和植被。
2. 视觉复核：助手逐张检查 40 张对照图，从严拒绝非机场硬质地面、建筑遮挡、明显色调/曝光不匹配、前景碎屑、阴影/来源背景残留等。本轮另拒绝 18 张。完整逐例判断及原因在 `code/reviews/v6_mar20_40_20260924.json`；筛选脚本要求 40 张都有明确判断，不能漏审默认通过。
3. 交付复验：独立脚本检查 20 张的图像/标签/mask 数量与哈希、训练来源、mask 外接框、原 MAR20 标注保留和新增 OBB 坐标；**20/20 通过结构校验**。通过结果的拼图又整体复看一次。

通过集各类新增目标：C-130 **4**、C-17 **4**、F-16 **4**、E-3 **3**、B-1B **2**、F-15 **2**、C-5 **1**。B-52 与 F/A-18 本轮没有达到从严筛选标准，因此**没有**强行放入；这说明交付集是质量优先的试批，不是均衡的最终训练集。

8999 看板的 `🔬 copy-paste` 分组新增两项 `✅ Copy-Paste v6·质检通过`，分别看对照与合成图/OBB。旧的未筛选 v6 项仍保留作失败分析，不应误当交付集。

## 边界

这批可以作为**视觉质量已筛、标签结构完整的小规模增强候选集**，但不等于已经证明对训练有益，也不能保证绝无争议样本。自动门只识别一种前景污染，**不是全自动视觉质量判别器**；在扩大到成千上万张之前，仍需提高自动语义/落点筛查能力并做抽检。当前标签仍是 MAR20 20 类体系；映射到指定 15 类另行处理。模型收益必须由固定验证集上的原始训练与增强训练对照实验确认。

## 复现

```bash
python3 code/synthesize_copypaste_v6.py --mode mar20 --count 40 --seed 20260925 --output /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v6_mar20_40_review
python3 code/curate_copypaste_v6.py --input /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v6_mar20_40_review --review code/reviews/v6_mar20_40_20260924.json --output /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v6_mar20_40_curated_v3
python3 code/verify_curated_copypaste_v6.py /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v6_mar20_40_curated_v3
```

复跑时输出目录需更换为新的空目录；脚本拒绝覆盖已有结果。
