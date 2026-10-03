# Copy-Paste v5：当前粘贴方法

**版本约定**：v4 是 MTARSI 全量抠图；v5 是把通过质量门的飞机前景贴入遥感场景的当前版本。

v5 以 [Simple Copy-Paste](https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html) 的 mask 粘贴为基础，加入面向机场图像的确定性位置选择；这部分是工程启发式，不是新训练的模型，也不是对 [InstaBoost](https://openaccess.thecvf.com/content_ICCV_2019/html/Fang_InstaBoost_Boosting_Instance_Segmentation_via_Probability_Map_Guided_Copy-Pasting_ICCV_2019_paper.html) 算法的完整复现。

1. **前景与背景**：从 v4 质量门通过的 MTARSI-fixed / INNAR Train 飞机透明图取前景；背景使用 LAE-1M 中有飞机标注的 DIOR Train 图。
2. **大小和位置**：参考目标图中已有飞机框的像素尺寸缩放；围绕已标注飞机试探候选位置，排除越界、碰撞已有标注、亮度或纹理差异过大的位置，再选得分最低的位置。固定随机种子，可复现。
3. **融合**：用前景原背景与目标位置的颜色中位数估计 RGB 偏移，最大调整 ±25；按 mask 粘贴，边缘羽化约 0.8 像素。**当前不生成投影阴影**，也不保证屋顶、草地等区域一定被排除。

当前 40 张样例是**未经人工筛选的算法原样输出**，在 8999 的“MTARSI→DIOR 粘贴预览 v5”入口审阅；代码为 [`code/paste_aircraft_preview.py`](../code/paste_aircraft_preview.py)。结果在 `/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_preview_fixed_v7_40/`。目录中的 `v7_40` 是此前试运行的内部命名，**项目版本号以此文档的 v5 为准**。这些样例尚未合并 DIOR 原标签与目标细分类别，不能直接作正式训练集。
