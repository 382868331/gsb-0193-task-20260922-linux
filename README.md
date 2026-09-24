# 复杂孔径的傅里叶衍射采样（Fraunhofer 模型）

纯 Python 标准库实现的可复用离线模块：对 ≤32×32 的复振幅孔径做二维离散傅里叶
变换得到远场，返回强度、物理坐标、总功率误差与中心索引；支持确定性圆孔 / 矩形孔
生成与居中零填充（≤64×64）。**不使用任何 FFT 包或第三方数值库**，DFT 为自行实现
的混合基 Cooley–Tukey（素数长度走 O(n²) 直接求和回退）。

本次迭代新增**有限波长、非负权重的相干模态入口** `fraunhofer_modes`：一次调用
可给 1–3 个 `CoherentMode`（各自孔径、波长、非负权重，至多 3 个不同波长），
每个模态独立 Fraunhofer 传播，再按 `I = Σ w_q·|U_q|²` 逐像元叠加**强度**；
跨波长、跨模态的复振幅永不相加（因此结果对象不提供合成 `field`，各模态复场保留
在 `mode_results` 中）。原有 `fraunhofer` 单模态入口与全部旧行为不变。

## 运行环境

- Python **3.14.7**（仅使用标准库 `cmath` / `math` / `dataclasses`）
- Linux 原生运行，安装 Python 后**无需安装任何其他依赖**，无需网络、账号、密钥、
  数据库、中间件、云 API 或 Docker
- 核心算法、测试、演示全部离线、确定性

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（37 个用例）
python demo.py                            # 固定输入演示，约 0.1 秒（上限 90 秒）
```

## 数学约定（固定契约）

- 输入孔径 `A` 为 N×N 方阵（1 ≤ N ≤ 32），**居中（shifted）布局**，像元间距
  `dx > 0`（米）。
- 前向 DFT 符号约定：

  ```
  S[k,l] = Σ_{m,n} a[m,n] · exp(−i·2π·(k·m + l·n)/M)
  U      = dx² / (wavelength · z) · S          # Fraunhofer 远场复振幅
  du     = wavelength · z / (M · dx)           # 观测面像元间距（米）
  ```

  强度 `I = |U|²`。
- 零填充：`pad=M`（N ≤ M ≤ 64）时把孔径居中嵌入 M×M 零阵；`du` 与全部物理坐标
  **始终按 M 重新计算**（填充改变观测间距）。`pad=None` 时 M=N。
- 偶数网格 fftshift 约定：索引 `p` 的物理坐标为 `x[p] = (p − M//2)·du`，
  即索引 `M//2` 恰为坐标 0（DC 像元）。偶数 M 时负方向比正方向多一个采样
  （x 从 −M/2·du 到 (M/2−1)·du）；奇数 M 时两侧对称。返回的 `center=(M//2, M//2)`。
- Parseval：`dx²·Σ|A|² = du²·Σ|U|²`。`power_error` 为**有符号相对误差**
  `(Pout−Pin)/Pin`；零输入时分母无定义，显式返回 `0.0`（此时 `input_power=0`）。

## 接口

```python
from fourier_diffraction import fraunhofer, circular_aperture, rectangular_aperture

r = fraunhofer(aperture, dx, wavelength, z, pad=None) -> FraunhoferResult
```

- `aperture`：可迭代的可迭代行，元素为 `int | float | complex`（自动转 complex），
  形状必须为方阵且非空。
- `dx, wavelength, z`：有限正实数（米、米、米）。
- `pad`：`None` 或严格 `int`，`N ≤ pad ≤ 64`。

`FraunhoferResult`（frozen dataclass，不可变）字段：

| 字段 | 含义 |
|---|---|
| `intensity` | M×M `float`，`|U|²`，居中布局，`intensity[i][j]` 对应 `(x[i], y[j])` |
| `field` | M×M `complex`，居中后的复场 U |
| `x`, `y` | 长度 M 的物理坐标（米），x 沿轴 0、y 沿轴 1 |
| `du` | 观测面像元间距（米，已按 M 更新） |
| `center` | 零坐标索引 `(M//2, M//2)` |
| `input_power` | `dx²·Σ|A|²` |
| `output_power` | `du²·Σ|U|²` |
| `power_error` | 有符号相对功率误差（零输入为 `0.0`） |
| `n`, `m` | 输入边长 / 变换边长 |

确定性孔径生成（均为纯函数，输出 `float` 的 0.0/1.0）：

```python
circular_aperture(n, radius, dx)
    # 含边界：hypot((i-n//2)dx, (j-n//2)dx) <= radius 的像元为 1
rectangular_aperture(n, width, height, dx)
    # 沿轴0宽 width、沿轴1高 height，含边界；半宽/半高闭区间判定
```

## 多波长 / 多模态非相干叠加（本次新增）

```python
from fourier_diffraction import fraunhofer_modes, CoherentMode

mode = CoherentMode(aperture, wavelength, weight=1.0)
r = fraunhofer_modes(modes, dx, z, pad=None) -> MultiModeResult
```

- `CoherentMode`（frozen dataclass）字段：`aperture`（N×N，约束同单模态入口）、
  `wavelength`（有限正实数，米）、`weight`（有限非负实数，默认 `1.0`；`bool` 按
  数值传入同样拒绝）。
- `modes`：1–3 个 `CoherentMode` 的可迭代对象（也接受生成器，会先物化为列表）。
  各模态必须共享边长 N；一次调用至多 **3 个不同波长**。`dx / z / pad` 全模态共用。
- 叠加规则（**只叠强度，绝不叠复振幅**）：

  ```
  I[p,q] = Σ_s weight_s · |U_s[p,q]|²
  ```

  每个模态各自调用 `fraunhofer` 独立传播，各自有自己的 `du_s = λ_s·z/(M·dx)` 与
  物理坐标；在共同的 M×M 像元索引网格上加权累加强度。不同波长的光互不相干，
  同波长的不同空间模态同样按非相干处理（权重即模态功率份额）。
- `MultiModeResult`（frozen dataclass）字段：

  | 字段 | 含义 |
  |---|---|
  | `intensity` | M×M `float`，`Σ_s w_s·|U_s|²`，居中布局 |
  | `fx`, `fy` | 长度 M 的**空间频率**坐标（cycles/m），`(p−M//2)/(M·dx)`；与波长无关，是叠加网格的公共轴 |
  | `center` | 零坐标索引 `(M//2, M//2)` |
  | `weights`, `wavelengths` | 校验后的各模态权重、波长（输入顺序） |
  | `input_power` | `Σ_s w_s·dx²·Σ|A_s|²` |
  | `output_power` | `Σ_s w_s·du_s²·Σ|U_s|²`；每模态各自满足 Parseval，故总量守恒到舍入 |
  | `power_error` | 有符号相对功率误差；总加权输入功率为 0 时显式为 `0.0` |
  | `mode_results` | 各模态独立的 `FraunhoferResult`（含各自的 `field / x / y / du`） |
  | `n`, `m` | 公共输入边长 / 变换边长 |

  结果对象**刻意没有 `field` 字段**：跨波长（以及非相干模态之间）不存在可相加的
  复振幅；需要单模态复场时取 `mode_results[s].field`。物理坐标换算：
  `mode_results[s].x[p] == wavelength_s · z · fx[p]`。
- 单模态退化：`fraunhofer_modes([CoherentMode(A, λ, 1.0)], dx, z, pad)` 的
  `intensity` 与 `fraunhofer(A, dx, λ, z, pad).intensity` 逐像元完全一致。
- 权重线性：所有权重乘常数 c，叠加强度与总功率整体乘 c（解析关系，测试按
  2 倍枚举核对）；权重为 0 的模态不产生任何贡献。

## 错误语义（非法输入不静默纠正）

- `TypeError`：物理量不是实数（`bool` 按数值传入同样拒绝）；`n` / `pad` 不是
  严格 `int`（**`bool` 即使是 `int` 子类也拒绝**）；孔径元素不是数值；孔径不可
  迭代或行不可迭代；`modes` 不可迭代、元素不是 `CoherentMode`（含 `bool`）；
  模态权重不是实数。
- `ValueError`：任何实数 / 复数元素为 `NaN` 或 `±Infinity`；`dx/wavelength/z ≤ 0`；
  `radius/width/height < 0`；空阵、非矩形、非方阵；N<1、N>32、M<N、M>64；
  模态数不在 1–3；不同波长超过 3 个；模态权重为负或非有限；各模态孔径边长不一致。
- 错误信息包含**可定位参数名**（如 `wavelength`、`aperture[i][j]`，多模态下为
  `modes[1].weight`、`modes[0].aperture[i][j]` 等）。
- `fraunhofer_modes` 在**任何传播之前**完成全部校验（孔径先全部物化解析）；函数
  为纯函数、无批次 / 有状态写操作，因此不存在部分变更；不存在搜索类算法，
  故无“预算耗尽”与“证明无解”之分（本契约的失败模式只有上面的参数拒绝）。

## 浮点误差口径（不以宽容差掩盖边界错误）

- DFT 与直接定义（逐项复指数求和，见测试内独立参考）逐 bin 比较：
  长度 1、2、3、5、7、8、12、16、31、32、48、64 上相对误差 < **1e-11**（按最大
  幅值归一）。
- Parseval 相对功率误差：N=1…32 实测量级 ≤ **1e-13**（demo 中为 −1.1e-14 量级），
  测试判据为 **1e-12**（这是纯累加舍入，不包含任何近似模型误差）。
- 矩形孔远场与**离散解析式**（有限等比级数闭式解，非 DFT 计算）逐像元比较
  < 1e-12（按全场最大幅值归一）；单像元常强度用精确相等 / 1e-15 判据。
- 坐标为整数倍 `du`，用 `1e-20` 绝对判据核对；DC 像元坐标严格为 `0.0`。

## 测试的独立性

`tests/test_fourier_diffraction.py` 内的期望值全部来自本文件自带的独立实现：
暴力 O(n²) DFT、矩形孔有限等比级数闭式解、手工 Parseval 求和。被测核心从不参与
生成期望值；正反 DFT 往返仅作为补充一致性测试。

覆盖：自研 DFT 定义一致性、单像元常强度与线性相位、矩形孔离散解析式、圆孔确定性
与对称性、Parseval 独立核功率、零填充功率不变与间距更新、相位斜坡导致的整数像元
平移（移位定理）、奇偶网格 fftshift 坐标约定、全部错误语义。

多模态新增覆盖（期望值同样不来自被测核心）：单模态退化逐 bin 与旧入口一致；
双波长单像元的**手算常数场**叠加与加权 Parseval；3 模态 3 波长在独立暴力 DFT
参考上逐像元一致；权重加倍全场加倍的线性关系；离轴像元证明“只叠强度”无干涉条纹
（并用各模态 `field` 演示错误的跨模态复场相加确会振荡）；公共 `fx` 轴与
`x_q = λ_q·z·fx`、`du ∝ λ` 关系；零权重 / 零孔径边界；frozen 不可变；模态数、
类型、权重、波长、孔径尺寸、公共物理量与 pad 的全部失败路径及可定位信息；
拒绝后重试结果不变（无半更新）。

## 设计取舍

- Fraunhofer（远场）模型是本库唯一物理模型；不实现 Fresnel / 角谱传播。
- 多模态叠加只输出强度（非相干叠加的物理定义）；即便各模态波长相同也不提供复场
  求和，避免把正交/非相干模态误当成相干光束。跨波长比较位置请用公共 `fx/fy` 轴。
- 各模态共用 N、`dx`、`z`、`pad`（共同像元网格是强度逐 bin 相加的前提）；波长与
  权重按模态独立。
- 居中嵌入与 fftshift 采用统一模运算映射（非负、不创建负索引），对奇偶 N、M 一致。
- DFT 输入规模很小（≤64，模态 ≤3），优先正确性与可读性；混合基 + 素数回退使 64
  长度变换仍是毫秒级，不为此引入第三方库。
- 接口只接受 Python 原生嵌套序列，返回原生 list，无 numpy 依赖。

## 已知限制

- 仅实现本题契约的数学子集：标量传播、均匀采样、单次传播、**至多 3 个模态与 3 个
  不同波长**；**不宣称**与任何行业标准光学软件兼容。
- 多模态结果无合成复场（设计如此）；不模拟色散介质、波长相关孔径响应或模态间
  相干性/部分相干（互相干矩阵）。
- 不做近场/倾斜因子/孔径内振幅包络以外的物理效应；`U` 的全局常数相位
  Fraunhofer 相位项按题面公式省略。
- 64 点素数回退路径只在非 2/3 因子长度出现（如 M 为素数，最大为 61），仍为
  O(n²)，在本题规模下耗时以毫秒计。
- 坐标与强度单位假定全部输入以米给出，库内不做单位换算。
