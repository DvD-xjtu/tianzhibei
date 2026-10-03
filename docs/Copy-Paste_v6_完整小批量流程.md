# Copy-Paste v6：完整小批量流程与结果

更新时间：2026-09-24。这里的版本按项目阶段命名：v4 是 MTARSI 全量抠图，v5 是首次粘贴预览，v6 是从合格前景到合成图、训练标签、自动校验和看板审阅的闭环。当前只跑了两组各 20 张，没有启动全量生成或模型训练。

## 输入与类别

- 前景：MTARSI-fixed 与 MTARSI-INNAR **Train** 的已抠图、自动质量门通过样本；INNAR Test/valid 不参与。C-5 不在原 15 类全量前景池中，暂用 v4 百例质量门通过的 5 张 C-5，后续扩量前需单独全量处理。
- 背景 A：LAE-FOD 的 DIOR train，含至少两个已有飞机框的机场图像。它的原标签是粗粒度 `airplane`，不能直接当成 MAR20 机型标签训练。
- 背景 B：MAR20 train，保留原有 20 类 OBB 标注，新增飞机沿用同一类 ID。只用 MAR20 train 的框统计尺度，未使用 test 图像或标注。
- 本轮 9 类：C-130、C-17、C-5、F-16、E-3、B-52、B-1B、F-15、F/A-18。每组的类数分别是 3、3、2、2、2、2、2、2、2；来源 fixed 4 张、INNAR Train 16 张。

## 算法实际步骤

1. 读取质量门通过的 RGBA 前景，按 alpha 裁切并在 ±15° 内旋转。旋转与缩放使用预乘 alpha，减少透明边缘的色晕。
2. 从 MAR20 train 的各类框面积分布求 P10/P50/P90。目标面积取该类中位数与当前背景飞机中位数的几何平均，再施加少量随机扰动，并裁剪到 P10–P90；过小或过大的尺寸拒绝。
3. 在现有飞机周围采样候选位置。拒绝出界、与任何原标签框相交、与同场景停机坪表面颜色差异过大、纹理/边缘过强或明显绿色植被的候选位置；从余下候选中择优。它是基于规则的机场场景 placement sampler，**不是**可学习的位置生成模型。
4. 对前景做有上限的 LAB 颜色偏移；仅当前景比局部背景锐利得多时轻微模糊；用软 alpha 边缘融合。没有使用 Poisson 融合，也没有生成投影阴影。因此个别前景颜色、透视和阴影仍可能显得不自然。
5. 保存合成图、合成实例 mask、原图/合成图对照、源文件与输出 SHA-256、配置、逐例清单和自动 QA 统计。DIOR 只输出新增目标的 HBB，原 DIOR 注释另存，标记 `train_ready=false`；MAR20 输出原 OBB 加合成目标 OBB，格式层面可用于 MAR20 同类任务，标记 `train_ready=true`。两者均不覆盖原始数据。
6. 独立校验脚本逐张核对文件哈希、mask 与框、类别 ID、坐标范围、背景/前景无重复、训练 split 和与原标注的碰撞。校验通过不等于视觉质量完全通过。

## 本轮实测

| 场景 | 生成 | 自动结构校验 | 产物定位 | 标签状态 |
| --- | ---: | ---: | --- | --- |
| DIOR train | 20/20 | 20/20 | `paste_v6_dior_20_r2` | 预览；原 `airplane` 标签未映射，不能直接训练 MAR20 |
| MAR20 train | 20/20 | 20/20 | `paste_v6_mar20_20_r2` | 保留原 OBB 并追加合成 OBB；尚未做模型收益验证 |

产物根目录均在 `/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/`。每个目录下有 `final/train/`、`comparison/train/`、`instance_masks/`、`manifest.jsonl`、`background_train.jsonl`、`generation_config.json` 与 `qa/`。8999 看板的 `🔬 copy-paste` 分组已加入 v6 的 DIOR/MAR20 原图对照和合成图，共四项，不修改共享前端样式。

本轮是**完整的小批量生成与校验**，不是已证明有效的训练增强。人工浏览拼图能看到部分飞机与场景的亮度、色调、阴影或朝向仍不完全一致；自动规则不能保证零 bad case。尤其 DIOR 缺少 20 类完整背景标签，若日后用于训练，必须先确定粗类飞机的 ignore/补标策略，不能把仅有新增框的标签误作完整训练标注。MAR20 合成数据也应先经人工抽检与基线/增强对照训练，再决定扩量。

与 `pipeline_design.md` 的生产门禁相比，本轮是 smoke test：程序先从 train 候选中自动选背景，再把实际选用的 20 张写为 `background_train.jsonl`；**并非**负责人事先冻结的六库训练背景清单。因此不能据此宣称六库全量方案已经就绪。

用相同随机种子、两个新输出目录分别生成首例时，合成图和标签的 SHA-256 均一致。看板重载后四个 v6 项目各能读取 20 张；MAR20 合成图项目共读出 149 个 OBB 框（原有框加 20 个新增框）。

## 复现与校验

代码：`code/synthesize_copypaste_v6.py`、`code/verify_copypaste_v6.py`。从项目根目录运行；输出目录须不存在或为空。

```bash
python3 code/synthesize_copypaste_v6.py --mode dior --count 20 --output /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v6_dior_20_r2
python3 code/synthesize_copypaste_v6.py --mode mar20 --count 20 --output /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v6_mar20_20_r2
python3 code/verify_copypaste_v6.py /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v6_dior_20_r2
python3 code/verify_copypaste_v6.py /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v6_mar20_20_r2
```

复跑时请换新目录名；不要向已有产物目录叠加写入。

后续已进行 **40 张 MAR20 候选 → 20 张质量筛选通过** 的独立试批；交付目录、自动/视觉筛选规则及逐例拒绝原因见 [《Copy-Paste v6：质量筛选与小批量交付》](Copy-Paste_v6_质量筛选与可用样本.md)。上文的 DIOR/MAR20 各 20 张仍是未筛选预览，不应与该交付批混淆。
