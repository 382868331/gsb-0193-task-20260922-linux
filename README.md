# 复杂孔径的傅里叶衍射采样（Fraunhofer 模型）

纯 Python 标准库实现的可复用离线模块：对 ≤32×32 的复振幅孔径做二维离散傅里叶
变换得到远场，返回强度、物理坐标、总功率误差与中心索引；支持确定性圆孔 / 矩形孔
生成与居中零填充（≤64×64）。**不使用任何 FFT 包或第三方数值库**，DFT 为自行实现
的混合基 Cooley–Tukey（素数长度走 O(n²) 直接求和回退）。

除单波长入口 `fraunhofer` 外，还提供**有限波长、非负权重的相干模态入口
`fraunhofer_modes`**：最多 3 个波长组、每组最多 3 个模态，每个模态用自己的
波长/孔径/权重独立传播，探测器上只叠加**强度**
`I = Σ_g Σ_t w[g][t]·|U[g][t]|²`，任何阶段都不跨波长（也不跨模态）相加复振幅。

## 运行环境

- Python **3.14.7**（仅使用标准库 `cmath` / `math` / `dataclasses`）
- Linux 原生运行，安装 Python 后**无需安装任何其他依赖**，无需网络、账号、密钥、
  数据库、中间件、云 API 或 Docker
- 核心算法、测试、演示全部离线、确定性

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（39 个用例）
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

## 多波长 / 多模态入口（非相干强度叠加）

```python
from fourier_diffraction import fraunhofer_modes

r = fraunhofer_modes(groups, dx, z, pad=None) -> MultimodeResult
```

物理约定（本迭代固定）：

- `groups` 为 1..3 个**波长组**，每个波长组携带 1..3 个**模态**（均为接口
  上限）。波长必须是有限正实数，且各组波长**两两不同**；同一物理波长的多个
  模态应放进同一个组。
- 每个模态有独立孔径 `A_m`（与 `fraunhofer` 相同的方阵规则，1..32）和
  **有限非负实权重** `w_m ≥ 0`（`bool` 拒绝，允许 0）。
- 每个模态按各自波长**独立**做一次普通 Fraunhofer 传播，得到自己的复场
  `U_m`、坐标与功率；探测器网格只累加加权强度：

  ```
  I[i][j] = Σ_g Σ_t w[g][t] · |U[g][t][i][j]|²
  ```

  **绝不跨波长相加复振幅**；同一波长下的不同模态也只叠加强度（不相干）。
  权重是朴素线性系数，库内**不做归一化**（w=2 即该模态强度翻倍）。
- 网格对齐：所有模态共享同一个 **DFT-bin 网格**（边长 M）。`pad=None` 时
  `M = max(N_m)`，否则 `max(N_m) ≤ pad ≤ 64`。每个波长仍有自己的物理间距
  `du_g = wavelength_g·z/(M·dx)`，记录在该模态自己的结果里；叠加发生在
  索引对齐的 bin 网格而非物理坐标上，各分量的 `x/y` 因此可以不同。

输入写法（三种等价形式可混用）：

```python
groups = [
    {"wavelength": 450e-9, "modes": [
        {"aperture": ap1, "weight": 1.0},   # 显式权重
        {"aperture": ap2},                  # 权重默认 1.0
    ]},
    (550e-9, [ap3, (ap4, 0.5), (ap5,)]),   # 组用 (波长, 模态列表)；裸孔径权重 1
]
r = fraunhofer_modes(groups, dx=1e-5, z=0.1, pad=32)
```

消歧规则：列表/元组形式的模态，仅当第二项是标量实数时才视为
`(aperture, weight)`；其余列表/元组一律按裸孔径网格解析（因此 N×N 的 list
孔径可以直接裸放，不会被当成包装）。

`MultimodeResult`（frozen dataclass）字段：

| 字段 | 含义 |
|---|---|
| `intensity` | M×M `float` 探测器强度，即各模态 `w·|U|²` 之和（居中布局） |
| `components` | 所有模态的扁平元组（按组、按模态顺序），每项是 `ModeComponent` |
| `wavelengths` | 各组波长元组（输入顺序） |
| `weights` | 各组权重元组的元组，与输入分组对齐 |
| `total_weight` | 全部权重之和（仅供参考；**不**用来除强度） |
| `detector_power` | `Σ_m w_m·du_m²·Σ|U_m|²`，叠加的功率预算（各模按自身物理间距） |
| `m` / `pad` | 公共变换边长 / 实际使用的变换边长（`pad=None` 时二者相同） |
| `center` | 公共零坐标索引 `(M//2, M//2)` |

`ModeComponent`（frozen dataclass）字段：`wavelength`、`weight`、
`result`（该模态完整的 `FraunhoferResult`，含 field/intensity/x/y/du/功率/误差）、
`weighted_power = weight·result.output_power`。每个模态各自满足 Parseval
（`result.power_error` 可逐模态检查，实测 ≤ 1e-13）。

**核验方式**（期望值不来自被测实现）：单模态 w=1 退化为普通 `fraunhofer`
（逐像元精确相等）；居中单像元 DFT 恒为 1，故多波长叠加为手算常数
`Σ w·(dx²/(λz))²`；权重满足精确线性与可加性（w=2 与两个 w=1 同孔径模态
给出同一强度）；一般网格用测试内独立暴力 DFT 逐模态计算后加权求和比对。

## 错误语义（非法输入不静默纠正）

- `TypeError`：物理量不是实数（`bool` 按数值传入同样拒绝）；`n` / `pad` 不是
  严格 `int`（**`bool` 即使是 `int` 子类也拒绝**）；孔径元素不是数值；孔径不可
  迭代或行不可迭代。
- `ValueError`：任何实数 / 复数元素为 `NaN` 或 `±Infinity`；`dx/wavelength/z ≤ 0`；
  `radius/width/height < 0`；空阵、非矩形、非方阵；N<1、N>32、M<N、M>64。
- 错误信息包含**可定位参数名**（如 `wavelength` 或 `aperture[i][j]`）。
- `fraunhofer_modes` 额外拒绝：`groups` 不是可迭代对象；波长组数不在 1..3；
  每组模态数不在 1..3；组/模态结构不是规定的映射或二元组；出现未知键或缺键；
  组波长为 0/负/非有限/非实、`bool`，或**重复波长**；模态权重为负、NaN、
  ±Infinity、非实或 `bool`；`pad < max(N_m)` 或 > 64。错误信息带可定位路径
  （如 `groups[1].modes[0] weight`、`groups[0].modes[0] aperture[i][j]`）。
  所有输入在任何传播发生前完成校验，且调用为纯函数，因此**不会留下半更新状态**。
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
- 多模态叠加：单模态退化与权重 w=0/1 用**精确相等**核对；权重线性用
  `1e-12·max(1, 峰值)` 判据（仅累加舍入）；3 波长 / 6 模态混合网格相对独立
  暴力 DFT 参考逐像元 < **1e-10**（按全场峰值归一，含多次加权累加）。

## 测试的独立性

`tests/test_fourier_diffraction.py` 内的期望值全部来自本文件自带的独立实现：
暴力 O(n²) DFT、矩形孔有限等比级数闭式解、手工 Parseval 求和、手算单像元常数。
被测核心从不参与生成期望值；正反 DFT 往返仅作为补充一致性测试。

覆盖：自研 DFT 定义一致性、单像元常强度与线性相位、矩形孔离散解析式、圆孔确定性
与对称性、Parseval 独立核功率、零填充功率不变与间距更新、相位斜坡导致的整数像元
平移（移位定理）、奇偶网格 fftshift 坐标约定、全部错误语义；以及多模态部分的
单模态退化（与 `fraunhofer` 精确一致）、三种等价输入写法、3×3 上限、默认网格取
最大孔径、各波长独立 `du`、独立暴力 DFT 的强度叠加核对、同波长反相场不发生相干
相消、权重线性与可加、零权重边界、网格/填充边界、全部结构与数值错误、失败不留
半更新状态。

## 设计取舍

- Fraunhofer（远场）模型是本库唯一物理模型；不实现 Fresnel / 角谱传播。
- 居中嵌入与 fftshift 采用统一模运算映射（非负、不创建负索引），对奇偶 N、M 一致。
- DFT 输入规模很小（≤64），优先正确性与可读性；混合基 + 素数回退使 64 长度变换
  仍是毫秒级，不为此引入第三方库。
- 接口只接受 Python 原生嵌套序列，返回原生 list，无 numpy 依赖。
- 多模态按题意只在强度域叠加（互强度矩阵的对角、非相干极限）；即使同一波长下的
  模态也不相干叠加复场，因此不需要模态间相位/交叉项。不同波长的强度在公共 DFT-bin
  索引网格上相加，物理坐标差异通过逐模态 `du/x/y` 保留，不做跨波长重采样。
- 单模入口与多模入口共用同一个内部传播函数 `_propagate`，二者数学路径完全一致。

## 已知限制

- 仅实现本题契约的数学子集：标量传播、均匀采样、单次传播；**不宣称**与任何行业
  标准光学软件兼容。
- 不做近场/倾斜因子/孔径内振幅包络以外的物理效应；`U` 的全局常数相位
  Fraunhofer 相位项按题面公式省略。
- 64 点素数回退路径只在非 2/3 因子长度出现（如 M 为素数，最大为 61），仍为
  O(n²)，在本题规模下耗时以毫秒计。
- 坐标与强度单位假定全部输入以米给出，库内不做单位换算。
- 多模态仅限本题边界：≤3 波长组、每组 ≤3 模态、单网格边长 ≤64；不支持波长连续
  谱积分、模态间相干交叉项、探测器光谱响应曲线或跨网格物理坐标插值；权重不自动
  归一化，总功率为各模态加权功率之和。
