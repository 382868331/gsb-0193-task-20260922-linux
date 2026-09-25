# 复杂孔径的傅里叶衍射采样（Fraunhofer 模型）

纯 Python 标准库实现的可复用离线模块：对 ≤32×32 的复振幅孔径做二维离散傅里叶
变换得到远场，返回强度、物理坐标、总功率误差与中心索引；支持确定性圆孔 / 矩形孔
生成与居中零填充（≤64×64）。**不使用任何 FFT 包或第三方数值库**，DFT 为自行实现
的混合基 Cooley–Tukey（素数长度走 O(n²) 直接求和回退）。
另含逆问题入口 `inverse_aperture.py`：由指定物理坐标上的**复振幅**观测，
在固定布尔支持 mask 上矩阵无关地求解 `||F a − b||² + lam·||a||²`（lam>0）
的正则反演，见下文专节。

## 运行环境

- Python **3.14.7**（仅使用标准库 `cmath` / `math` / `dataclasses`）
- Linux 原生运行，安装 Python 后**无需安装任何其他依赖**，无需网络、账号、密钥、
  数据库、中间件、云 API 或 Docker
- 核心算法、测试、演示全部离线、确定性

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（37 个用例）
python demo.py                            # 正向传播固定输入演示，约 0.1 秒（上限 8 秒）
python demo_inverse.py                    # 逆问题固定输入演示，约 0.2 秒（上限 90 秒）
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

## 逆问题：由复振幅观测反推有限孔径（`inverse_aperture.py`）

在正向传播之上新增的逆问题入口。观测是**若干指定物理坐标上的复振幅**
（不是只有强度），孔径支持区由固定布尔 mask 给出，求解

```
min_a  ||F a − b||² + lam·||a||²,        lam 严格 > 0
```

其中 `F` 是与正向模型完全一致的 Fraunhofer 算子（复用同一像元中心
`ξ_m=(m−n//2)·dx`、相位约定 `exp(−i2π(xξ+yη)/(λz))` 与物理比例
`dx²/(λz)`），在 DFT 网格点上与 `fraunhofer(...)` 的场逐 bin 重合（有测试）。

### 可复现使用路径

```python
import inverse_aperture as inv

mask = [[True, True], [True, False]]          # N×N 严格 bool，True 为未知量
observations = [
    (1.0e-4, -2.0e-4, 0.0123 - 0.004j),        # (x, y, 复振幅)，单位米
    (1.0e-4, -2.0e-4, 0.0123 - 0.004j),        # 重复观测 = 重复测量，保留不去重
    (-3.0e-4, 1.0e-4, -0.0021 + 0.009j),
]
result = inv.invert_aperture(
    mask, observations,
    dx=10e-6, wavelength=0.5e-6, z=0.10,
    lam=1e-6,                 # 严格 > 0
    tol=1e-10, max_iter=None, # 默认 1000；到达上限不算收敛
)
result.aperture               # N×N 复孔径，mask 外严格为 0j
result.residual               # 真实残差 F a − b（每个观测一项，重复观测重复出现）
result.residual_norm          # ||F a − b||
result.objective              # ||F a − b||² + lam·||a||²（对最终解显式重算）
result.normal_residual_norm   # ||(F*F + lam I) a − F* b||（显式重算）
result.converged, result.stop_reason, result.iterations
```

`stop_reason` 取值：`"converged"` / `"max_iter_reached"` / `"zero_rhs"` /
`"no_observations"` / `"empty_support"`；只有 `"converged"`（及三个平凡
情形）时 `converged=True`，**到达迭代上限一律 `converged=False`**。

### 实现要点

- 正向算子 `ApertureOperator.forward` 与共轭转置 `ApertureOperator.adjoint`
  分别实现；只缓存逐轴可分离相位表（`n_obs × N` 两个），**从不建立**
  `n_obs × n_unknown` 的稠密傅里叶矩阵。
- 正规方程 `(F*F + lam I) a = F* b` 用矩阵无关共轭梯度迭代求解；
  残差、目标、正规方程残差均对最终迭代点**显式重算**，不取自 CG 递推量。
- 重复观测作为重复测量保留，残差与目标中各计一次，不去重、不改权重。
- 观测数可以少于未知量数（`lam>0` 保证正规方程正定）；空观测列表得到
  零孔径，全 `False` mask 得到零孔径且残差恰为 `−b`。

### 失败示例（真实触发）

```python
inv.invert_aperture(mask, observations, 10e-6, 0.5e-6, 0.10, 0.0)
# ValueError: lam must be > 0, got 0.0     —— lam=0 时欠定正规方程奇异

inv.invert_aperture(mask, observations, 10e-6, 0.5e-6, 0.10, 1e-6, max_iter=1)
# converged=False, stop_reason='max_iter_reached' —— 上限到达不当作收敛
```

`demo_inverse.py` 现场合成观测（独立暴力求和，非被测算子）、反演并重新
投影，打印上述两类失败边界。

### 正则解不是唯一真实孔径

欠采样时数据不能唯一确定孔径；`lam>0` 在拟合与能量之间折中，返回的是
**最小（残差+lam·能量）孔径**——与数据相容的无穷多孔径之一，而非"那个"
真实孔径。`demo_inverse.py` 用真实输出展示恢复孔径与合成真值的偏差。
本库不做强度相位恢复（phase retrieval），观测必须含复振幅（幅度+相位）。

### 逆问题错误语义

- `TypeError`：mask 元素不是严格 `bool`；观测不是 `(x, y, amplitude)` 三元
  组或振幅不是数值；`max_iter` 不是严格 `int`（`bool` 同样拒绝）。
- `ValueError`：`lam ≤ 0`、NaN/Inf；观测坐标为 NaN/Inf；mask 非方、为空、
  N>32；`tol ≤ 0`；`max_iter < 1`。错误信息含可定位参数名
  （如 `observations[0]`、`mask[i][j]`、`lam`）。

## 错误语义（非法输入不静默纠正）

- `TypeError`：物理量不是实数（`bool` 按数值传入同样拒绝）；`n` / `pad` 不是
  严格 `int`（**`bool` 即使是 `int` 子类也拒绝**）；孔径元素不是数值；孔径不可
  迭代或行不可迭代。
- `ValueError`：任何实数 / 复数元素为 `NaN` 或 `±Infinity`；`dx/wavelength/z ≤ 0`；
  `radius/width/height < 0`；空阵、非矩形、非方阵；N<1、N>32、M<N、M>64。
- 错误信息包含**可定位参数名**（如 `wavelength` 或 `aperture[i][j]`）。
- 计算为纯函数，无批次 / 有状态写操作，因此不存在部分变更；不存在搜索类算法，
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

`tests/test_inverse_aperture.py` 的期望值同样来自文件内独立实现：逐项构造的
显式稠密傅里叶矩阵 + 复数高斯消元解正规方程、手工复内积。伴随一致性
（`<Fx, r> == <x, F*r>`，标准复内积）、小孔径显式矩阵比对、不完整采样、
零观测、空 mask、不同 lambda、重复观测保留、迭代上限不算收敛均有覆盖；
与正向模型在网格点上的一致性、满采样恢复仅作为补充。

覆盖：自研 DFT 定义一致性、单像元常强度与线性相位、矩形孔离散解析式、圆孔确定性
与对称性、Parseval 独立核功率、零填充功率不变与间距更新、相位斜坡导致的整数像元
平移（移位定理）、奇偶网格 fftshift 坐标约定、全部错误语义。

## 设计取舍

- Fraunhofer（远场）模型是本库唯一物理模型；不实现 Fresnel / 角谱传播。
- 居中嵌入与 fftshift 采用统一模运算映射（非负、不创建负索引），对奇偶 N、M 一致。
- DFT 输入规模很小（≤64），优先正确性与可读性；混合基 + 素数回退使 64 长度变换
  仍是毫秒级，不为此引入第三方库。
- 接口只接受 Python 原生嵌套序列，返回原生 list，无 numpy 依赖。

## 已知限制

- 仅实现本题契约的数学子集：标量传播、均匀采样、单次传播；**不宣称**与任何行业
  标准光学软件兼容。
- 不做近场/倾斜因子/孔径内振幅包络以外的物理效应；`U` 的全局常数相位
  Fraunhofer 相位项按题面公式省略。
- 64 点素数回退路径只在非 2/3 因子长度出现（如 M 为素数，最大为 61），仍为
  O(n²)，在本题规模下耗时以毫秒计。
- 坐标与强度单位假定全部输入以米给出，库内不做单位换算。
- 逆问题只解 Tikhonov 正则的线性最小二乘；不处理观测噪声建模、lam 自动
  选取、非负/稀疏约束，也不做仅强度观测的相位恢复。CG 收敛速度依赖观测
  几何，病态几何时可能在迭代上限内达不到 tol（此时如实报告
  `max_iter_reached`）。
