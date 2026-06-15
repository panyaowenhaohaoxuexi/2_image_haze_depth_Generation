# ASM + Depth Anything V2 + 红外天空掩膜批量加雾工具

这个工具用于对 FLIR 等道路/驾驶场景可见光图像批量生成物理大气散射模型（ASM）加雾图。每张输入图只估计一次深度、一次大气光、一次天空掩膜，然后输出多档能见度结果，保证同一张图的轻雾/中雾/浓雾空间分布一致，仅浓度不同。

## 环境安装

推荐在你的 Anaconda `CoA` 环境中运行：

```powershell
conda activate CoA
pip install -r requirements.txt
```

如果 `CoA` 环境里已经安装了匹配 CUDA 的 `torch`，可以先保留现有版本，只安装缺失包。

## 运行方式

```powershell
conda activate CoA
python batch_asm_haze.py `
  --vis_dir F:\path\to\vis `
  --ir_dir F:\path\to\ir `
  --out_dir F:\path\to\out_haze
```

默认输出：

- `原文件名_V200.jpg`
- `原文件名_V100.jpg`
- `原文件名_V50.jpg`
- `out_haze/debug/原文件名_depth.png`
- `out_haze/debug/原文件名_sky_mask_soft.png`
- `out_haze/debug/原文件名_V100_t_final.png`

修改能见度档位：

```powershell
python batch_asm_haze.py --vis_dir vis --ir_dir ir --out_dir out --visibilities 300 150 75
```

如果红外天空掩膜在某批图上误分了路面或反光区域，可以先关闭天空掩膜，改用纯深度透射率：

```powershell
python batch_asm_haze.py --vis_dir vis --ir_dir ir --out_dir out --disable_sky_mask
```

关闭后不会执行 `t_final = (1 - sky_mask_soft) * t_depth + sky_mask_soft * t_sky` 的天空强制混合，实际等价于 `t_final = t_depth`。

## 模型权重

默认模型 ID：

```text
depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
```

可选模型：

- `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`：室外/驾驶，速度快，推荐首选。
- `depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf`：室外/驾驶，精度更高，显存要求更大。
- `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`：室内场景。
- `depth-anything/Depth-Anything-V2-Small-hf`：非 Metric 相对深度版，需设置 `--no-is_metric_depth` 并配合 `--depth_scale` 使用。

指定模型：

```powershell
python batch_asm_haze.py `
  --vis_dir vis `
  --ir_dir ir `
  --out_dir out `
  --model_name_or_path depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
```

也可以把 `--model_name_or_path` 指向本地 HuggingFace 模型目录。

## metric 深度与 depth_scale

脚本顶部 `CONFIG` 中默认：

```python
"IS_METRIC_DEPTH": True
"DEPTH_SCALE": 80.0
```

两种模式必须区分：

- 使用 Depth Anything V2 metric（绝对深度）版时，模型输出单位已是米：`depth_m = 模型输出`。此时脚本强制 `depth_scale=1.0 effective`，忽略 `--depth_scale`。
- 使用 Depth Anything V2 相对深度版时，模型输出没有真实米制单位：`depth_m = relative_depth * depth_scale`。此时默认 `depth_scale=80.0`，需要按数据标定。

相对深度版标定方法：

1. 在图中选择一个已知真实距离的参照物，例如路牌、车辆或标定点，记为 `known_distance_m`。
2. 查看该位置附近模型输出的相对深度均值，记为 `relative_d`。
3. 计算 `depth_scale = known_distance_m / relative_d`。
4. 用新的比例运行：`--no-is_metric_depth --depth_scale 你的比例`。

## 可见光与红外配对规则

默认同名配对：

```text
vis/abc.jpg -> ir/abc.jpg
```

如果你的命名是：

```text
xxx_rgb.jpg -> xxx_ir.jpg
```

打开 `batch_asm_haze.py`，修改 `find_paired_ir_path()` 中的配对逻辑。函数内已经保留了对应注释示例。

红外缺失或读取失败时，脚本会打印 warning，并自动回退为“纯深度阈值天空处理”，不会中断批处理。

## 物理模型

脚本严格使用：

```text
I(x) = J(x) * t(x) + A * (1 - t(x))
t(x) = exp(-beta * d(x))
beta = 3.912 / V
```

- `J`：清晰可见光图。
- `I`：加雾图。
- `A`：暗通道先验估计的大气光。
- `d(x)`：Depth Anything V2 输出并转换后的米制深度。
- `V`：气象能见度，单位米。

天空区透射率使用红外天空软掩膜平滑混合：

```text
t_final = (1 - sky_mask_soft) * t_depth + sky_mask_soft * t_sky
```

默认 `t_sky=0.07`。如果天空雾色过重，可调大到 `0.1`；如果天空仍太清晰，可调小到 `0.05`。

## Debug 验收

每张图都会在 `debug` 文件夹输出：

- `*_depth.png`：深度可视化，检查是否近处浅/远处深。
- `*_sky_mask_soft.png`：红外天空软掩膜，检查天空边界是否平滑。
- `*_V100_t_final.png`：透射率图，检查近处透射率高、远处和天空透射率低。

验收重点：

- `V200` 雾最轻，`V100` 居中，`V50` 最浓。
- 同一张图三档结果的空间雾分布一致，仅整体浓度递增。
- 天空区域接近雾色，交界处没有硬边和破碎伪影。
- 近处保留更多清晰纹理，远处更容易被雾遮挡。
