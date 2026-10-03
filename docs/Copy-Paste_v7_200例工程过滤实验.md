# Copy-Paste v7：200 例纯工程过滤实验

本轮目标是验证“代码能自动挡掉多少明显风险样本”，不是宣称数据已通过人工质检。候选图沿用 v6 的合成流程；v7 新增独立的工程质量门。**本批 200 张没有经过助手视觉检查、人工挑选或逐图改判**，留给你在 8999 看板审阅。

## 数据与结果

- 前景：MTARSI-fixed 与 MTARSI-INNAR 已通过抠图的飞机；背景：MAR20 **train**。未使用 MAR20 验证/测试图作背景。
- 本轮只测原始目标清单中当前可映射 MAR20 的 8 类：C-130、C-17、F-16、E-3、B-52、B-1B、F-15、F/A-18；每类生成 25 张候选。C-5 不在用户原清单，本轮不生成。其余目标类的映射与可用前景尚未补齐。
- 候选 **200/200** 结构校验通过；工程过滤 **56/200 通过（28.0%）**，**144/200 拒绝（72.0%）**；通过集 **56/56** 文件与 OBB 标签结构校验通过。结构校验和规则通过均不等于视觉质量通过。

| 类别 | 候选 | 工程通过 | 通过率 |
| --- | ---: | ---: | ---: |
| C-130 | 25 | 9 | 36% |
| C-17 | 25 | 10 | 40% |
| F-16 | 25 | 9 | 36% |
| E-3 | 25 | 3 | 12% |
| B-52 | 25 | 5 | 20% |
| B-1B | 25 | 8 | 32% |
| F-15 | 25 | 7 | 28% |
| F/A-18 | 25 | 5 | 20% |

## 过滤逻辑

过滤器只读取图像、实例 mask、原背景与合成记录，不读取人工审阅标签。任一条件触发即拒绝：

1. **源前景**：mask 内与源图周边背景颜色相近的像素占比过高（疑似带入背景）；mask 次级连通域面积占比过高（疑似碎片）。
2. **粘贴位置**：局部纹理、植被、边缘密度或颜色差异过大，避免贴在明显不合适的地面/复杂结构上。
3. **融合效果**：飞机相对原背景对比度太低，或亮度、饱和度差异太大；灰度背景上的彩色前景也拒绝。

阈值固定在 [`filter_copypaste_v7.py`](../code/filter_copypaste_v7.py) 的 `GATES`，每张的指标及**全部**触发原因在 `qa/decisions.jsonl`。下表按第一条触发原因归类，合计 144；一张图可能同时触发多条规则。

| 首要拒绝原因 | 数量 |
| --- | ---: |
| 位置纹理过强 | 55 |
| 粘贴亮度不匹配 | 26 |
| 飞机与背景对比不足 | 18 |
| 位置边缘过密 | 13 |
| 源前景疑似带背景 | 11 |
| mask 碎片 | 8 |
| 位置植被过多 | 8 |
| 饱和度不匹配 | 2 |
| 位置颜色不匹配 | 2 |
| 灰度背景上出现彩色前景 | 1 |

阈值用上一批 40 张开发样本作过小规模回归：代码放行 12 张，均属于此前视觉通过的 21 张；此前视觉拒绝的 19 张均被代码挡住，另有 9 张误拒。**这是调参开发集表现，不能当作本批 200 张的精度或“零坏例”保证**。本批只报告工程通过率，实际可用率须待你审阅。

## 文件与看板

- 全部候选：`/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v7_mar20_200_candidates/`
- 工程通过集：`/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v7_mar20_200_engineering_pass/`
- 全量逐例判定：通过集下的 `qa/decisions.jsonl`；汇总与阈值：`qa/summary.json`；通过集 `train/images`、`train/labels` 为合成图及 MAR20 20 类 YOLO-OBB 标签；`manifest.jsonl` 保留来源追踪。
- 8999 看板的 `🔬 copy-paste` 组新增“v7·全部候选 原图/合成图”“v7·工程通过 原图/合成图”“v7·工程通过 合成图/OBB”三项。前两项适合审阅合成效果，后一项检查最终训练图及标签。

复现命令（原输出目录已有结果；重跑须指定**新目录**，不要覆盖）：

```bash
cd /home/wangduyun/aircraft-copypaste-lae
python3 code/synthesize_copypaste_v6.py --mode mar20 --count 200 --seed 20260926 --classes C-130,C-17,F-16,E-3,B-52,B-1B,F-15,F/A-18 --output /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/NEW_CANDIDATES
python3 code/verify_copypaste_v6.py /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/NEW_CANDIDATES
python3 code/filter_copypaste_v7.py --input /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/NEW_CANDIDATES --output /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/NEW_ENGINEERING_PASS
python3 code/verify_curated_copypaste_v6.py /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/NEW_ENGINEERING_PASS
```

当前标签仍采用 MAR20 的 20 类索引；目标 15 类对齐和训练收益验证是后续工作。你审阅后给出坏例 ID，可用于下一版规则回归，而不是回头人工挑选本版通过集。
