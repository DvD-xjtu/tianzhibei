# MTARSI → LAE-1M Copy-Paste Pipeline 实现设计（v1）

**日期：**2026-09-23  
**状态：**实现设计；首轮 smoke test 参数为待验证默认值  
**目标：**以 MTARSI 飞机分类裁剪图生成可信前景，粘贴到 LAE-1M 的训练背景，形成可追溯的 MAR20 细粒度检测增强样本。

## 1. 范围与已知依据

首轮仅使用 MTARSI-fixed 与 MTARSI-INNAR 的 Train 前景；INNAR Test/valid 不纳入首轮。背景只能来自经负责人确认的 `manifests/background_train.jsonl`，不得使用验证或测试图。原始数据保持只读。

| 统一类别 | MAR20 ID（依据当前项目讨论，实施前与正式类表校验） |
|---|---:|
| B-1B | 9 |
| B-52 | 7 |
| C-130 | 1 |
| C-17 | 2 |
| C-5 | 3 |
| E-3 | 6 |
| F-15 | 12 |
| F-16 | 4 |
| F/A-18 | 15 |

A-10、F-35、E-2 只留待扩展 taxonomy；`C-135` 不映射为 `KC-135`，`KC-767` 不映射为 `KC-10/KC-135`。不能通过名称近似自动推断型号。

**已完成的先导实验：**SAM2.1-tiny 全图自动分割 50 图，平均 3.07 秒/图，峰值显存 2.73 GiB，44/50 有候选 mask，人工初筛约 22/50 通过。因此改用提示式分割，并对其单独做实测验收。INNAR F-16 抽查 60/395 全为近俯视，方位角丰富；旋转默认 ±15°，不以镜像补视角差异。

背景候选飞机框面积占图比例的 P5/P50/P95（百分数）如下。此为候选库临时统计，不能替代最终选中背景子集统计。

| 来源 | 图像尺寸概况 | 飞机框数 | P5 / P50 / P95 |
|---|---|---:|---|
| DIOR | 800×800 | 717 | 0.034 / 2.855 / 14.168% |
| DOTAv2 | 1024×1024 | 7,469 | 0.043 / 0.486 / 6.246% |
| FAIR1M | 600×600 | 25,250 | 0.331 / 1.156 / 3.867% |
| NWPU VHR-10 | 多尺度，约 500–1000 px | 226 | 0.318 / 0.758 / 1.944% |
| RSOD | 多尺度，常见约 1044×915 | 2,149 | 0.050 / 0.282 / 1.076% |
| xView | 1024×1024 | 281 | 0.067 / 0.107 / 0.342% |

## 2. 阶段及交付

| 阶段 | 输入 | 关键处理 | 输出与停机条件 |
|---|---|---|---|
| 0 背景清单 | LAE-1M 候选训练图及原标签 | 核对 split、路径、尺寸、标签格式与可放置区域；冻结清单 | `background_train.jsonl`；未冻结时不能运行合成 |
| 1 前景池 | MTARSI-fixed / INNAR Train | 显式类别映射、SAM2 提示分割、规则质检及抽检 | 候选/通过 manifest、mask、QA；抽检未通过不得批量合成 |
| 2 合成 | 冻结的背景清单与通过的前景池 | 条件化尺度、受限放置、旋转、碰撞、融合、标签生成 | 合成图片、标签与逐实例记录 |
| 3 审计 | 阶段 0–2 产物 | 联系表、计数、类别与背景分布、泄漏与标签检查 | `qa/summary.json`、检查记录；人工审核 smoke test |

建议输出根目录：`/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/`。

```text
<output_root>/
├── source_masks/<source_id>.png
├── manifests/
│   ├── background_train.jsonl
│   ├── foreground_candidates.jsonl
│   ├── foreground_accepted.jsonl
│   ├── synth_records.jsonl
│   └── generation_config.json
├── synth_yolo/train/images/
├── synth_yolo/train/labels/
├── qa/contact_sheets/
├── qa/rejected_foregrounds/
├── qa/summary.json
└── logs/
```

## 3. 配置与类别映射

配置分为 `configs/copypaste_v1.yaml` 和 `configs/class_mapping.yaml`；绝对路径在目标服务器配置，不写死到算法模块。示例：

```yaml
version: lae1m_mtarsi_copypaste_v1
seed: 20260923
taxonomy:
  mode: MAR20_20
  enabled_classes: [B-1B, B-52, C-130, C-17, C-5, E-3, F-15, F-16, F/A-18]
  official_class_table: /path/to/verified_mar20_classes.yaml
foreground:
  sources: [MTARSI-fixed, MTARSI-INNAR]
  innar_splits: [Train]
  max_accepted_per_class: 300
  min_source_size: 80
segmentation:
  model_type: sam2.1_tiny
  checkpoint: /path/to/checkpoint.pt
  prompt_mode: center_box_plus_center_point
  center_box_ratio: [0.15, 0.15, 0.85, 0.85]
  multimask_output: true
quality_filter:
  min_mask_area_ratio: 0.01
  max_mask_area_ratio: 0.80
  max_border_touch_ratio: 0.05
  max_connected_components: 2
  min_bbox_fill_ratio: 0.15
synthesis:
  max_images: 100
  max_new_instances_per_image: 3
  max_synth_instances_per_class: 50
  rotation_deg_range: [-15, 15]
  max_hbb_iou_existing: 0.05
  max_hbb_iou_new: 0.05
  max_placement_attempts: 100
  scale_strategy: background_conditioned
  blend_method: feather
  feather_px: 2
  random_full_image_fallback: false
background:
  manifest_path: manifests/background_train.jsonl
output:
  root: /data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1
```

类别映射须列出所有允许项与显式拒绝项；未列出的源类默认拒绝。以下名称须以实际目录扫描核对后冻结，拼写保留源数据原样。

```yaml
MTARSI-fixed:
  B-1: B-1B
  B-52: B-52
  C-130: C-130
  C-17: C-17
  C-5: C-5
MTARSI-INNAR:
  B-1_Lancer: B-1B
  B-52_Stratofortress: B-52
  C-130_Hercules: C-130
  C-17_Globemaster: C-17
  C-5_Galaxy: C-5
  E-3_Sentry: E-3
  F-15_Eagle: F-15
  F-16_Falcon: F-16
  F-18_Hornit: F/A-18
reject: [A-10, A-10_Thunderbolt, E-2, E-2_Hawkeye, F-35, F-35_JSF, C-135, C-135_Stratolifter, KC-767_Tanker]
```

## 4. Stage 0：背景清单与标签语义

每条背景记录必须引用一张**已确认 train split** 的图，路径存在，尺寸与标签解析一致；以数据集名和稳定图像 ID 唯一标识，重复图及同源验证/测试重合须检查。记录原始标注和可用的放置区域/依据。统计冻结子集的图像尺寸、目标数及飞机框面积比，再拟合采样先验。

现有飞机标签识别：DIOR `0: airplane`；DOTAv2 `0: plane`；FAIR1M ID `0,1,2,3,4,7,8,9,10,13,34`（对应 A220、A321、A330、A350、ARJ21、Boeing737、Boeing747、Boeing777、Boeing787、C919、other airplane）；NWPU `0: airplane`；RSOD `0: aircraft`；xView `0: Small Aircraft`、`1: Cargo Plane`。xView `48: Aircraft Hangar` **不是飞机**。ID 的含义必须根据每个数据集自己的类表读取，不能跨库复用。

**训练标签的关键前置条件：**上述 LAE 粗类或民航机型标签与 MAR20 细类 ID 不同。原背景标签可以逐字保留在独立来源标签文件，但不能原样复制到 MAR20 YOLO 标签后追加合成标签。若要产出可直接训练的 MAR20 图集，必须明确并实现原目标的映射或 ignore 策略，并验证训练器确实支持 ignore；否则将合成标签写在独立文件，产物标记为 `not_train_ready`。当背景中存在无法精确映射的飞机而训练器不支持 ignore 时，跳过该图，避免把真实飞机当负例。非飞机原目标同样按正式训练任务的类表处理。这一门禁必须在 Stage 2 前确认。

## 5. Stage 1：前景池

按 `source_dataset / source_split / source_class_raw` 扫描，统一类别后做白名单过滤；原图及来源路径记录在 manifest。对每张合格裁剪图运行 SAM2.1-tiny：中心点 `(w/2,h/2)`、中心框 `[0.15w,0.15h,0.85w,0.85h]`，保留多候选及模型分数。中心点或固定中心框只是初始假设，抽样对比点、框、组合提示，失败时保留拒绝原因。

每个候选 mask 计算面积占比、连通域数、最小外接 HBB 的填充率、中心偏移、是否触四边及边界接触量。`border_touch_ratio` 定义为**mask 中位于图像最外侧一像素边框的像素数 / mask 像素数**，防止不同实现混用边框长度和面积。默认拒绝空 mask、面积比不在 `[0.01,0.80]`、边界接触比 `>0.05`、连通域 `>2`、填充率 `<0.15` 的候选；记录各指标及所有拒绝原因。可用大连通域占比和中心偏移排序候选，阈值通过人工抽检后校准；这些规则不能证明飞机轮廓完整，也不能代替人工验收。源图触边或包含多架飞机应标注并拒绝。

每类随机抽检至少 30 个通过的 mask（不足则全检），同时检查一定比例被拒 mask 以估计误拒率；记录是否完整包含机身与双翼、是否漏翼、含阴影或背景、是否多目标。通过的 mask 以无损 PNG 存储，不改动原图。保存模型 checkpoint 的哈希及 prompt/规则配置快照。

## 6. Stage 2：合成策略

### 6.1 尺度

依据**冻结背景子集**按来源构建飞机 HBB 面积比的经验分布，并记录样本量及分位数。抽样时按背景来源取面积比例 `r`，经过经 smoke test 校准的类别组调整（`fighter`：F-15/F-16/F/A-18；`large_transport`：C-130/C-17/C-5；`large_bomber`：B-1B/B-52；`awacs`：E-3），限制到该来源可靠分位区间。候选六库统计只能用作初始参考，不能把单库的几个百分位当各类别真实分布；样本很少的来源需限制尾部或回退到审定范围。

令背景宽高为 `W,H`，目标 HBB 面积为 `rWH`。基于**旋转后的前景 mask HBB** 宽高 `w_f,h_f` 求等比缩放 `s = sqrt(rWH/(w_f h_f))`，缩放后重新计算 mask HBB 和实际面积比，越界或过小则重采样。不要以 mask 面积冒充 HBB 面积。

### 6.2 旋转和位置

旋转 `[-15°,15°]`，用透明画布防止裁掉翼尖，图像与 mask 同变换；不默认镜像。优先在已有飞机邻域采样，其次使用经过确认的机场/停机坪 ROI；现有大目标仅在其语义确认为可放置区域时可作邻域参考。默认禁用全图随机回退，找不到可靠区域则跳过背景并记录原因。对已知飞机、其他标注目标、禁止区域及本次已放置对象做边界和碰撞检查；新实例须完整在图内，HBB 与现存及新增框的 IoU 均 `<0.05`。IoU 门槛无法保证没有遮挡或地理不合理，smoke test 需人工查验。限制尝试次数，失败不产出伪阳性记录。

### 6.3 融合和标注

采用前景 mask 构造 RGBA，仅边缘 feather 约 2 px；最终检测框从**实际粘贴后的非零实例 mask** 重算，裁剪并归一化为 YOLO `class_id cx cy w h`，检查每个数值在 `[0,1]`、宽高为正。若配置保存 OBB，仅保存在审计记录，不能混进 HBB 文本。依据 §4 的标签语义门禁生成最终训练标签；保存原背景标注的独立引用或副本、追加实例的独立记录，绝不无条件合并不同 taxonomy 的数字 ID。

一张输出图允许 1–3 个新增实例，记录每一个实例的前景来源、实际尺度和位置。输出命名使用稳定背景 ID 与合成索引，并进行原子写入，避免中断留下半张图或半份标签。

## 7. Manifest schema（JSONL，每行一个 JSON 对象）

全部路径采用绝对路径或明确的 `dataset_root` 相对路径，统一 `schema_version: 1`，UTF-8 编码。`source_id`、`background_id` 和 `synth_id` 在本次构建内唯一；bbox 统一为绝对像素坐标 `xyxy`，左上包含、右下不包含，坐标系与图像相同。空值明确用 `null`。

### 7.1 `background_train.jsonl`：每行一张背景

```json
{"schema_version":1,"background_id":"fair1m_000123","dataset":"FAIR1M","split":"train","image_path":"/abs/bg.jpg","label_path":"/abs/bg.txt","width":600,"height":600,"label_format":"yolo_hbb","class_map_version":"fair1m_classes_v1","aircraft_boxes_xyxy":[[120,180,185,230]],"placement_roi_path":null,"source_hash_sha256":"..."}
```

`aircraft_boxes_xyxy` 可由解析器依据来源类表生成，但写入清单时须能复核；所有标注框另外交给碰撞模块，不能只检查飞机。`placement_roi_path` 若有值，须记录 ROI 制作方式及版本。清单冻结后记录文件 SHA-256；合成运行时校验哈希，禁止悄然更换背景集合。

### 7.2 `foreground_candidates.jsonl`：每行一个源图候选结果

```json
{"schema_version":1,"source_id":"mtarsi_innar__Train__F-16_Falcon__000345","source_dataset":"MTARSI-INNAR","source_split":"Train","source_class_raw":"F-16_Falcon","source_class_unified":"F-16","target_class_id":4,"image_path":"/abs/fg.jpg","mask_path":"/abs/mask.png","width":256,"height":256,"hbb_xyxy":[31,42,210,188],"mask_area_ratio":0.22,"border_touch_ratio":0.0,"connected_components":1,"bbox_fill_ratio":0.37,"sam_score":0.91,"accepted":true,"reject_reasons":[]}
```

未产生 mask 时 `mask_path` 与 `hbb_xyxy` 为 `null`，填入原因。`foreground_accepted.jsonl` 仅复制通过项，可添加人工复核状态和质量排序；进入合成前仅取满足当前验收策略的记录。

### 7.3 `synth_records.jsonl`：每行一张输出图，内含各实例

```json
{"schema_version":1,"synth_id":"fair1m_000123__20260923__0001","synth_image_path":"/abs/out.jpg","synth_label_path":"/abs/out.txt","background_id":"fair1m_000123","background_dataset":"FAIR1M","background_image_path":"/abs/bg.jpg","background_label_path":"/abs/bg.txt","background_manifest_sha256":"...","label_policy":"mar20_with_verified_ignore","train_ready":true,"seed":3261792340,"instances":[{"foreground_source_id":"mtarsi_innar__Train__F-16_Falcon__000345","source_split":"Train","target_class_id":4,"target_area_ratio":0.008,"actual_hbb_area_ratio":0.0079,"scale_factor":0.72,"rotation_deg":8.3,"paste_center_xy":[512,438],"hbb_xyxy":[487,413,537,463],"max_iou_existing":0.013,"max_iou_new":0.0}],"blend_method":"feather","accepted":true,"reject_reasons":[]}
```

示例数值仅说明字段结构，须由实际图像和计算结果填写。被拒尝试写单独审计日志或在记录中用 `accepted:false` 并设输出路径为 `null`，不得假装有已生成图像。`label_policy` 必须与实际训练器配置相符。

## 8. 模块与 CLI

```text
copypaste_pipeline/
├── config.py, taxonomy.py, io_utils.py, geometry.py, yolo_utils.py
├── foreground/
│   ├── scan_mtarsi.py, sam2_segmenter.py
│   ├── quality_filter.py, build_foreground_pool.py
├── background/
│   ├── build_manifest.py, parse_manifest.py, label_parser.py, stats.py
├── synth/
│   ├── scale_sampler.py, rotation_sampler.py, placement_sampler.py
│   ├── collision.py, blender.py, generator.py
├── qa/
│   ├── contact_sheet.py, summary.py, validators.py
└── cli/
    ├── build_backgrounds.py, build_foregrounds.py
    ├── synthesize.py, smoke_test.py
```

```bash
python -m copypaste_pipeline.cli.build_backgrounds --config configs/copypaste_v1.yaml
python -m copypaste_pipeline.cli.build_foregrounds --config configs/copypaste_v1.yaml --class-mapping configs/class_mapping.yaml
python -m copypaste_pipeline.cli.smoke_test --config configs/copypaste_v1.yaml --max-images 100
python -m copypaste_pipeline.cli.synthesize --config configs/copypaste_v1.yaml
```

`build_backgrounds` 应先产生候选清单，经过 split、标签及 ROI 检查并由负责人冻结后才作为 Stage 2 的输入；上述命令只是接口设计。

## 9. 可复现性与验收

固定源文件排序、随机数生成器和配置快照；**不要使用 Python 内置 `hash()` 派生种子**，它可能随进程改变。局部种子可取 `SHA-256(f"{global_seed}|{background_id}|{synth_index}")` 的前 4 字节，再记录种子、软件版本、SAM2 checkpoint 哈希、清单哈希与输出图/标签哈希。重跑同配置应比较 manifest、坐标、类别和图像校验和；存在 GPU 非确定性时说明具体差异。

首轮上限 100 张图；每类人工检查至少 30 个通过 mask 和 30 个粘贴实例（不足则全查）。若 100 张内无法覆盖每类 30 个合成实例，另做分阶段抽查，**不要把“每类 30 张”和“总量 100 张”同时当硬门槛**。检查完整轮廓、背景泄漏、合理位置与尺度、碰撞、标签可视化及源 split；统计每类候选/接受率与失败原因、每背景库使用量、目标面积比分布、处理时间和人工错误率。`qa/contact_sheets/` 保存原图/叠加 mask/抠图，以及背景/合成图/框对照；`qa/summary.json` 包含这些计数和配置、清单哈希。

**进入 2,000–5,000 张首轮增强集的条件：**前景抽检通过、场景与标签抽检通过、训练标签语义门禁落实、背景清单冻结、可复现回放通过；扩量规模由各类实例上限、背景量和训练消融决定，不以 MAR20 train 图数机械对齐。

## 10. 实现前待确认的具体输入

1. 冻结的 `background_train.jsonl` 及每库原始类表、标签格式、split 核查结果和可放置区域策略。
2. MAR20 官方类别 ID 表与背景原标注的处理方式，尤其是粗类飞机的 ignore/过滤是否得到训练器支持。
3. 提示式 SAM2 在有代表性样本上的人工通过率；据此调整 prompt、质量阈值及按类接受上限。
4. 冻结背景子集的尺度分布与每类尺度修正量，首轮可先用保守范围试跑，再以 QA 结果校准。
