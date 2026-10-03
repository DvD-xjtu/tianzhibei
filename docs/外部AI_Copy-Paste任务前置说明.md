# 外部 AI 任务前置说明：MTARSI 飞机前景 Copy-Paste 至 LAE-1M

更新日期：2026-09-23。本文是交给外部 AI 设计并实现 Copy-Paste 算法前的事实边界和交付契约；请以此为准，不要修改或移动原始数据。

## 1. 任务目标

目标是将两版 MTARSI 中已经裁剪出的、带型号分类标签的飞机图像作为**前景素材**，以合理的实例级 Copy-Paste 方式合成到 LAE-1M 的大范围遥感场景中，形成新的训练样本。预期收益是补充飞机型号、姿态、背景与尺度组合，改善模型对战斗机及其他军机**具体型号**的识别能力。

MAR20 不作为本项目的粘贴背景或前景来源；它是带真实旋转框的独立飞机检测数据集，计划参与联合训练，并作为评估基准。其官方 test 集不得参与素材筛选、尺寸分布拟合、调参或训练。

本任务要生成的是**离线合成数据集**，不是训练时随机在线增强；每张合成图、每个新增实例、每个 mask 和每次随机选择均须可追溯与复现。

## 2. 数据角色与绝对路径

| 角色 | 数据集与位置 | 可做什么 | 不可做什么 |
|---|---|---|---|
| 前景源 | MTARSI-fixed：`/data3/tianzhibei/datasets/MTARSI-fixed/` | 读取分类目录中的飞机裁剪图；以目录名获取类别 | 不得就地写 mask、改名、删除或覆盖图片 |
| 前景源 | MTARSI-INNAR：`/data3/tianzhibei/datasets/MTARSI-INNAR/extracted/MTARSI-INNAR/{Train,Test,valid}/` | 同上；以末级目录名获取类别与原始 split | 不得把其 Test/valid 当作模型验证集；若它们用作外部前景素材，必须在清单中显式记录 |
| 被增强对象 / 背景源 | LAE-1M：`/data2/tianzhibei/LAE-1M/LAE-FOD/` | 从**预先冻结的训练图清单**读取遥感场景并复制为合成图底图 | 不得从待验证/测试图取背景；不得改原始图和标签 |
| 联合训练及评估 | MAR20：`/data3/tianzhibei/datasets/MAR20/` | 读取 `yolo_obb/train` 用于训练；按独立协议评估 | 官方 `yolo_obb/test` 绝不参与合成或模型选择 |
| 看板代码 | `/data2/tianzhibei/dataset-viewer/`（8999 端口） | 合成完成后登记并查看数据质量 | 本阶段不要修改运行中的服务或 `datasets.json` |

LAE-1M 候选背景子集及本机现有格式如下；这些只是候选，外部 AI 不可自行把所有文件视为训练集，应由 `background_train.jsonl` 决定实际可用图片。

| 背景子集 | 图像目录 | 标签目录 | 现有标签格式 | 本机图像/标签文件数（快照） |
|---|---|---|---|---:|
| DIOR | `DIOR/JPEGImages-trainval` | `DIOR/yolo_labels` | YOLO HBB：`cls cx cy w h` | 11,725 / 7,632 |
| DOTAv2 | `DOTAv2/images` | `DOTAv2/yolo_labels` | YOLO HBB | 17,480 / 13,906 |
| FAIR1M | `FAIR1M/images` | `FAIR1M/yolo_labels` | YOLO HBB | 64,147 / 47,324 |
| NWPU VHR-10 | `NWPU VHR-10/images` | `NWPU VHR-10/yolo_labels` | YOLO HBB | 650 / 395 |
| RSOD | `RSOD/images` | `RSOD/yolo_labels` | YOLO HBB | 976 / 701 |
| xView | `xview/train_images_1024_05` | `xview/yolo_labels` | YOLO HBB | 21,100 / 12,266 |

`HRSC2016/yolo_labels` 存在 289 个标签文件，但当前 `HRSC2016/images` 下没有可用图像；在路径修复并完成单独核验前，**不要把 HRSC2016 作为背景输入**。

## 3. 前景数据的实际状态

两版 MTARSI 都是**分类裁剪图**，不是已具有实例轮廓的检测数据：目录层级就是类别标签，未发现 HBB、OBB、polygon 或 alpha mask 标注。它们视觉上是以飞机为中心的裁剪图，但不能据此假设整张图都是飞机。

因此外部 AI 必须将“从分类裁剪图得到可信飞机前景”作为算法的第一阶段：

1. 对每张候选图生成飞机前景 mask（可使用通用分割模型、显著性分割、形态学后处理或它们的组合）。
2. 输出 mask、最小外接 HBB、最小旋转外接框（OBB）以及质量指标。
3. 过滤 mask 为空、多主体、严重触边、明显包含大块原机场背景/邻机、面积异常或形状异常的样本；不得在无法得到可信 mask 时退化为“整张裁剪图直接粘贴”。
4. 抽样保存原图、mask、抠图和粘贴结果，供人工检查。

### 前景类别（名称即语义，不可猜测合并）

MTARSI-fixed 当前实际可读的图片目录只有 12 类：`A-10`、`B-1`、`B-2`、`B-29`、`B-52`、`C-130`、`C-135`、`C-17`、`C-5`、`Commercial-2engine`、`Commercial-4engine`、`E-2`。目录中的图片数为 984；`COUNT.txt` 标注的 9,144 是数据集元信息，和当前实际落盘文件数不一致，算法必须以实际扫描结果及清单为准。

MTARSI-INNAR 有 44 个类别及 `Train/Test/valid` 三个分类 split；与目标型号直接相关的目录包括：`A-10_Thunderbolt`、`B-1_Lancer`、`B-52_Stratofortress`、`C-130_Hercules`、`C-135_Stratolifter`、`C-17_Globemaster`、`E-2_Hawkeye`、`E-3_Sentry`、`F-15_Eagle`、`F-16_Falcon`、`F-18_Hornit`、`F-35_JSF`。另有 `KC-767_Tanker`，但它不是 KC-10 或 KC-135。

严禁错误映射：`C-135` ≠ `KC-135`，`KC-767` ≠ `KC-10`，`F-18_Hornit` 可归一为 F/A-18，`B-1` 可归一为 B-1B，`B-52` 可归一为 B-52H。MTARSI 中不存在 F-12/YF-12A 和直升机。

## 4. MAR20 的作用、格式与标签映射

MAR20 共有 3,842 张约 800×800 图像，官方分为 1,331 张 train 和 2,511 张 test；原始 OBB XML 在 `Annotations/Oriented Bounding Boxes/`，已转换的 YOLO-OBB 在：

```text
/data3/tianzhibei/datasets/MAR20/yolo_obb/
├── train/images/  train/labels/     # 1,331 张
└── test/images/   test/labels/      # 2,511 张
```

YOLO-OBB 一行是 `class_id x1 y1 x2 y2 x3 y3 x4 y4`，坐标为 0–1 归一化四角点。其类别 id 与型号的固定映射为：

```text
0 A1=SU-35        1 A2=C-130       2 A3=C-17       3 A4=C-5
4 A5=F-16         5 A6=TU-160      6 A7=E-3        7 A8=B-52
8 A9=P-3C         9 A10=B-1B      10 A11=E-8      11 A12=TU-22
12 A13=F-15      13 A14=KC-135    14 A15=F-22     15 A16=F/A-18
16 A17=TU-95     17 A18=KC-10     18 A19=SU-34    19 A20=SU-24
```

MAR20 可覆盖的 MTARSI 同型号前景有 C-130、C-17、F-16、E-3、B-52、B-1B、F-15、F/A-18；它不含 A-10、F-35、E-2、F-12/YF-12A 或直升机。若训练目标是“具体型号”，需事先冻结一个统一标签空间：MAR20 原生 20 类与 MTARSI 扩展类不能仅靠数字 class id 拼接。

## 5. 必须先冻结的训练标签策略

外部 AI 可以实现数据合成，但**不得自行决定语义标签策略**。以下三种策略应由项目负责人选择一种并写入 `configs/class_mapping.yaml` 后才运行全量生成：

1. **MAR20-20 类主任务（推荐首轮）**：仅粘贴能准确映射到上述 20 个 MAR20 类的 MTARSI 前景；生成标签使用 MAR20 id。A-10、F-35、E-2 等只能留作后续扩展实验。
2. **统一扩展型号空间**：定义“MAR20 20 类 + MTARSI 新类”的新 taxonomy；训练、验证和指标都按此新空间配置。MAR20 没有的类只能在 MTARSI 合成训练集衡量，不能报告为 MAR20 AP。
3. **两阶段任务**：第一阶段只学“aircraft”检测，第二阶段用型号分类器区分机型。此时 LAE-1M 原有的泛飞机标注可用作第一阶段，但不能伪装成细粒度型号标签。

最重要的规则：LAE-1M 原有的 `airplane/plane` 等泛类框**不能**凭背景数据集名称或视觉猜测而赋成 C-17、F-16 等具体型号；只有 MTARSI 粘贴生成的实例才带其已知细粒度型号。

## 6. 外部 AI 的实现输入、输出与不可变规则

### 输入

- `configs/class_mapping.yaml`：由负责人冻结的来源类别 → 训练类别映射。
- `manifests/background_train.jsonl`：唯一允许作为合成底图的背景清单；每行至少包括 `dataset`、`image_path`、`label_path`、`split=train`、`image_id`。
- MTARSI 的上述原始目录。
- 可选的 `manifests/foreground_allowlist.jsonl`：限定前景来源与原始 split；若不存在，先只建立候选池与质检报告，不应直接做全量合成。

### 输出（统一写到 `/data3`，不写回原始数据）

```text
/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/
├── source_masks/<source_id>.png
├── manifests/
│   ├── foreground_candidates.jsonl
│   ├── foreground_accepted.jsonl
│   ├── synth_records.jsonl
│   └── generation_config.json
├── synth_yolo/
│   └── train/{images,labels}/
├── qa/
│   ├── contact_sheets/
│   ├── rejected_foregrounds/
│   └── summary.json
└── logs/
```

背景数据当前是 YOLO HBB，因此首个可训练交付默认使用 `synth_yolo/train/labels/*.txt`，每一行严格为 `class_id cx cy w h`（归一化 HBB）。原背景标签应被无损复制，再追加粘贴实例标签。若算法同时保存 OBB，可另建 `synth_yolo_obb/`，但不要把 9 列 OBB 标签混入上述 HBB 标签目录。

每条 `synth_records.jsonl` 至少记录：输出图路径、背景图绝对路径及其数据集、背景清单 id、前景原图路径、MTARSI 版本和原始 split、原始/统一类别、mask 路径、缩放比例、旋转角、粘贴中心、融合方式、随机种子、接受/拒绝理由、与已有框最大 IoU。

### 不可变规则

- 原始目录只读；不使用 `--clean` 或任何删除操作指向原始数据。
- 合成训练图只能来自 `background_train.jsonl`；验证/测试图零接触。
- 不得以整张 MTARSI 分类裁剪图作为前景替代物；必须有通过质检的 mask。
- 粘贴后的实例必须完整在图内，并和原有框及本轮新框做碰撞检测；推荐 HBB IoU < 0.05，若提供 OBB 则同时报告 OBB IoU。
- 每个合成版本使用显式 seed、冻结输入清单、版本化配置；不能在未记录的随机状态下覆盖旧版本。
- 合成图文件名必须与原背景图区分，例如 `<background_id>__cp__<seed>__<index>.jpg`，避免覆盖。

## 7. 建议算法能力与验收标准

算法选择由外部 AI 决定，但交付应具备以下能力：可信前景分割、按目标分辨率/尺寸分布缩放、任意角度旋转、自然融合、避开已有目标、标签同步更新、可复现记录和 QA 导出。可参考已有的旋转 Copy-Paste 基础实现：`/root/tianzhibei/algorithm/rtmdet_obb/scripts/synth_copypaste.py`；它适用于已有 OBB 实例，不能不经改造直接读取 MTARSI 分类图。

最小验收分两步：

1. **Smoke test**：每个启用的 MTARSI 类生成少量前景候选和不超过 100 张合成图；交付 contact sheet、拒绝原因统计、标签合法性检查和随机复现结果。
2. **全量生成前审查**：人工检查每类至少 30 个 mask 与至少 30 张粘贴结果；确认没有验证/测试背景泄漏、没有类别错映射、没有标签出界或重叠异常后才扩展规模。

## 8. 8999 看板预留接口（生成后才执行）

8999 看板扫描 YOLO HBB 或 YOLO-OBB 的同名标签。首批合成数据通过验收后，再向 `/data2/tianzhibei/dataset-viewer/datasets.json` 增加条目；本次外部 AI 的算法实现不应改动这个运行中服务。

HBB 首版可登记为：

```json
{
  "id": "lae1m-mtarsi-cp-v1",
  "name": "LAE-1M + MTARSI Copy-Paste v1",
  "root": "/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/synth_yolo",
  "format": "yolo",
  "names": ["以冻结的 class_mapping.yaml 为准"],
  "splits": [
    {"name": "train", "images": "train/images", "labels": "train/labels", "label_prefix": ""}
  ]
}
```

登记后在看板点击“重扫”或调用 `POST /api/reload`。看板仅用于查看图片、框与问题样本；mask 和前后对比仍以 `qa/` 目录的审计材料为准。
