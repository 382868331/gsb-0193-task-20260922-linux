# 复杂孔径的傅里叶衍射采样（Fraunhofer 模型）与复振幅观测反演

纯 Python 标准库实现的可复用离线模块，两部分：

1. **正向**（`fourier_diffraction.py`）：对 ≤32×32 的复振幅孔径做二维离散傅里叶
   变换得到远场，返回强度、物理坐标、总功率误差与中心索引；支持确定性圆孔 / 矩形孔
   生成与居中零填充（≤64×64）。**不使用任何 FFT 包或第三方数值库**，DFT 为自行实现
   的混合基 Cooley–Tukey（素数长度走 O(n²) 直接求和回退）。
2. **逆问题**（`inverse_aperture.py`）：在固定布尔支持 mask 上，从若干指定物理坐标
   处的**复振幅**观测（不是只有强度）反推复孔径，最小化
   `‖F a − b‖² + λ‖a‖²`（λ 严格大于 0）。正向算子与共轭转置分别实现、复用正向模型
   的像元中心 / 相位 / 物理比例；正规方程用**矩阵无关共轭梯度**迭代求解，不建立任意
   大的稠密傅里叶矩阵。

## 运行环境

- Python **3.14.7**（仅使用标准库 `cmath` / `math` / `dataclasses`）
- Linux 原生运行，安装 Python 后**无需安装任何其他依赖**，无需网络、账号、密钥、
  数据库、中间件、云 API 或 Docker
- 核心算法、测试、演示全部离线、确定性

## 准确命令（可复现使用路径）

```bash
./.venv/bin/python -m unittest discover -s tests -v   # 必要测试（36 个用例）
./.venv/bin/python demo.py                            # 固定输入演示，现场计算 < 1 秒（上限 90 秒）
```

`demo.py` 依次执行：正向 Fraunhofer 计算（32×32 圆孔，零填充到 64×64）→ 逆问题
（合成观测 → 反演 → 重新投影 → 两个不同 λ 的对比）→ 两个真实触发的拒绝边界。

## 失败示例（README 即代码，均可直接复现）

```python
import fourier_diffraction as fd, inverse_aperture as ia

fd.fraunhofer([[1.0]], 1e-5, 5e-7, 0.1, pad=True)
# TypeError: pad must be an int, got bool: True   （bool 不是合法整数参数）

ia.invert_aperture([(0.0, 0.0, 1+1j)], [[True]], 1e-5, 5e-7, 0.1, 0.0)
# ValueError: lam must be > 0, got 0.0            （λ 必须严格大于 0）

ia.invert_aperture([(0.0, 0.0, 1+1j)], [[1]], 1e-5, 5e-7, 0.1, 1.0)
# TypeError: mask[0][0] must be a bool, got int: 1 （mask 元素必须是严格 bool）
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

## 逆问题接口（inverse_aperture）

```python
from inverse_aperture import (
    invert_aperture, forward_operator, adjoint_operator, InverseResult,
)

res = invert_aperture(observations, mask, dx, wavelength, z, lam,
                      tol=1e-10, max_iterations=None) -> InverseResult
```

- **观测模型**：`observations` 为 `(x, y, value)` 三元组的可迭代对象——观测面物理
  坐标（米）与该处的**复振幅**观测值。可为空；**重复观测按重复测量保留**（同一三元
  组出现 k 次即加权 k 次），不去重。正向算子复用原像元中心、相位与物理比例：

  ```
  (F a)(x, y) = dx²/(λz) · Σ_{(i,j)∈support} a[i,j] · exp(−i2π(x·ξ_i + y·η_j)/(λz))
  ξ_i = (i − n//2)·dx,  η_j = (j − n//2)·dx
  ```

  在 DFT 网格点 `x = k·du`（`du = λz/(M·dx)`）上与 `fraunhofer` 的输出逐点一致，
  因此观测可直接取自 `FraunhoferResult.field / x / y`。
- **支持区**：`mask` 为 N×N 严格 `bool` 方阵（1 ≤ N ≤ 32），True 像元为未知量
  （行主序排列）；全 False 合法，解为零孔径。未知数系数顺序为 mask 的行主序扫描。
- **目标函数**：`J(a) = ‖F a − b‖² + lam·‖a‖²`，`lam` 有限且**严格 > 0**（λ=0
  直接 `ValueError`，因为欠采样时正规方程不再正定）。
- **求解器**：对正规方程 `(F*F + λI) a = F* b` 做矩阵无关共轭梯度；正向算子
  （`forward_operator`）与共轭转置（`adjoint_operator`）分别实现，核按轴可分离为
  两个逐观测指数向量，**从不建立** n_obs×n_unknowns 的稠密傅里叶矩阵。收敛判据为
  `‖F*b − (F*F+λI)a‖ ≤ tol·‖F*b‖`；`max_iterations=None` 时上限为
  `max(32, min(4096, 2·n_unknowns))`。
- **停止原因**：`stop_reason="converged"` 表示达到容差；`"max_iterations"` 表示
  到达迭代上限——**不算收敛**（`converged=False`），返回的诊断量仍是该迭代点的
  真实值。

`InverseResult`（frozen dataclass）字段：

| 字段 | 含义 |
|---|---|
| `aperture` | N×N 复孔径，mask 外严格为 `0j` |
| `residual` | 真实数据残差 `F a − b`，逐观测（含重复），复数列表 |
| `residual_norm` | `‖residual‖` |
| `objective` | 由返回解直接重算的 `‖F a − b‖² + lam‖a‖²` |
| `normal_residual` | 直接重算的正规方程残差 `‖F*F a + lam a − F* b‖`（非 CG 递推估计） |
| `stop_reason` | `"converged"` / `"max_iterations"` |
| `iterations` | 实际 CG 迭代次数 |
| `converged` | `stop_reason == "converged"` |
| `n_observations` / `n_unknowns` | 观测数（含重复）/ 支持像元数 |
| `lam` | 实际使用的正则权重 |

**正则解不是唯一真实孔径**：`lam > 0` 使目标函数的极小点唯一，但该极小点是正则化
估计而非“真实孔径”——不同 λ 给出不同解（λ 增大 → 孔径能量减小、数据残差增大），
欠采样时亦有多个孔径都能解释观测。本库不做强度相位恢复，也不接在线仪器。

## 错误语义（非法输入不静默纠正）

- `TypeError`：物理量不是实数（`bool` 按数值传入同样拒绝）；`n` / `pad` /
  `max_iterations` 不是严格 `int`（**`bool` 即使是 `int` 子类也拒绝**）；孔径元素
  不是数值；孔径 / mask 不可迭代或行不可迭代；mask 元素不是严格 `bool`。
- `ValueError`：任何实数 / 复数元素为 `NaN` 或 `±Infinity`；`dx/wavelength/z ≤ 0`；
  `radius/width/height < 0`；空阵、非矩形、非方阵；N<1、N>32、M<N、M>64；
  `lam ≤ 0`、`tol ≤ 0`、`max_iterations < 1`；观测三元组元素个数不为 3；
  系数 / 观测值个数与支持像元数 / 观测数不符；观测数超过 1 000 000（防御性上限）。
- 错误信息包含**可定位参数名**（如 `wavelength`、`aperture[i][j]`、
  `mask[i][j]`、`observations[k] x`）。
- 计算为纯函数，无批次 / 有状态写操作，因此不存在部分变更；迭代求解的“预算耗尽”
  与收敛明确分开：`stop_reason="max_iterations"` 不是收敛，也不伪装成收敛。

## 浮点误差口径（不以宽容差掩盖边界错误）

- DFT 与直接定义（逐项复指数求和，见测试内独立参考）逐 bin 比较：
  长度 1、2、3、5、7、8、12、16、31、32、48、64 上相对误差 < **1e-11**（按最大
  幅值归一）。
- Parseval 相对功率误差：N=1…32 实测量级 ≤ **1e-13**（demo 中为 −1.1e-14 量级），
  测试判据为 **1e-12**（这是纯累加舍入，不包含任何近似模型误差）。
- 矩形孔远场与**离散解析式**（有限等比级数闭式解，非 DFT 计算）逐像元比较
  < 1e-12（按全场最大幅值归一）；单像元常强度用精确相等 / 1e-15 判据。
- 坐标为整数倍 `du`，用 `1e-20` 绝对判据核对；DC 像元坐标严格为 `0.0`。
- 逆问题：CG 解与**独立显式稠密矩阵求解**（测试内自建核矩阵 + 复数高斯消元）在
  小孔径上逐项比较 < 1e-9（按解的最大幅值归一）；伴随一致性
  `Σ(Fa)·conj(b) = Σ a·conj(F*b)` 相对误差 < 1e-12；从 `fraunhofer` 网格观测
  反演（λ=1e-12）与真实孔径的最大偏差 < 1e-6（按真实孔径最大幅值归一）。

## 测试的独立性

`tests/test_fourier_diffraction.py` 内的期望值全部来自本文件自带的独立实现：
暴力 O(n²) DFT、矩形孔有限等比级数闭式解、手工 Parseval 求和。被测核心从不参与
生成期望值；正反 DFT 往返仅作为补充一致性测试。

覆盖：自研 DFT 定义一致性、单像元常强度与线性相位、矩形孔离散解析式、圆孔确定性
与对称性、Parseval 独立核功率、零填充功率不变与间距更新、相位斜坡导致的整数像元
平移（移位定理）、奇偶网格 fftshift 坐标约定、全部错误语义；逆问题的复内积伴随
一致性、小孔径显式稠密矩阵比对、不完整（欠定）采样、零观测、空 mask、不同 λ 的
残差–能量权衡、重复观测加权、迭代上限不算收敛、与正向 Fraunhofer 的往返恢复。

## 设计取舍

- Fraunhofer（远场）模型是本库唯一物理模型；不实现 Fresnel / 角谱传播。
- 居中嵌入与 fftshift 采用统一模运算映射（非负、不创建负索引），对奇偶 N、M 一致。
- DFT 输入规模很小（≤64），优先正确性与可读性；混合基 + 素数回退使 64 长度变换
  仍是毫秒级，不为此引入第三方库。
- 接口只接受 Python 原生嵌套序列，返回原生 list，无 numpy 依赖。
- 逆问题为矩阵无关实现：每次算子作用 O(n_obs·n_unknowns) 次复乘（可分离指数
  因子），CG 迭代次数 ≤ 未知量个数（精确算术意义下）；内存 O(n_obs·N)，不随
  观测数建立稠密核矩阵。

## 已知限制

- 仅实现本题契约的数学子集：标量传播、均匀采样、单次传播；**不宣称**与任何行业
  标准光学软件兼容。
- 不做近场/倾斜因子/孔径内振幅包络以外的物理效应；`U` 的全局常数相位
  Fraunhofer 相位项按题面公式省略。
- 64 点素数回退路径只在非 2/3 因子长度出现（如 M 为素数，最大为 61），仍为
  O(n²)，在本题规模下耗时以毫秒计。
- 坐标与强度单位假定全部输入以米给出，库内不做单位换算。
- 逆问题仅求解线性最小二乘 + Tikhonov 正则的复孔径估计；**不做**仅强度观测的
  相位恢复，不接在线仪器 / 实时数据流。观测坐标为任意指定物理坐标，但正向模型
  仍是同一 Fraunhofer 采样模型，不宣称超分辨率或带限外推能力。
