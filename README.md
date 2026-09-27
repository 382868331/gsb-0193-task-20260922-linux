# 复杂孔径的傅里叶衍射采样（Fraunhofer 模型）

纯 Python 标准库实现的可复用离线模块：对 ≤32×32 的复振幅孔径做二维离散傅里叶
变换得到远场，返回强度、物理坐标、总功率误差与中心索引；支持确定性圆孔 / 矩形孔
生成与居中零填充（≤64×64）。**不使用任何 FFT 包或第三方数值库**，DFT 为自行实现
的混合基 Cooley–Tukey（素数长度走 O(n²) 直接求和回退）。

## 运行环境

- Python **3.14.7**（仅使用标准库 `cmath` / `math` / `dataclasses`）
- Linux 原生运行，安装 Python 后**无需安装任何其他依赖**，无需网络、账号、密钥、
  数据库、中间件、云 API 或 Docker
- 核心算法、测试、演示全部离线、确定性

## 准确命令

```bash
python -m unittest discover -s tests -v   # 必要测试（34 个用例）
python demo.py                            # 固定输入演示，约 0.5 秒（上限 8 秒）
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

### 探测像素能量（面积积分）

```python
from fourier_diffraction import integrate_intensity, pixel_energy

e = pixel_energy(aperture, dx, wavelength, z, i, j, pad=None,
                 rtol=1e-6, atol=0.0, min_depth=2, max_depth=12,
                 max_evaluations=200000) -> IntegrationResult
e2 = integrate_intensity(aperture, dx, wavelength, z, x0, x1, y0, y1,
                         rtol=1e-6, ...) -> IntegrationResult
```

- `pixel_energy`：采样点 `(i, j)` 对应的 `du x du` 探测像元（以
  `((i-M//2)du, (j-M//2)du)` 为中心的方块）实际接收的能量，即连续
  Fraunhofer 强度在像元面积上的积分，**不是** `intensity[i][j] * du**2`。
- `integrate_intensity`：任意矩形 `[x0, x1] x [y0, y1]`（米）上的强度积分，
  用于把相邻像素合并后与同一总区域互相核对。
- 被积函数是**连续远场**（孔径直接求和，与 DFT bin 完全同号约定），不是对
  采样网格的插值。

`IntegrationResult`（frozen dataclass）字段：

| 字段 | 含义 |
|---|---|
| `energy` | 区域内的强度积分（像素接收能量） |
| `error_estimate` | 保守绝对误差估计（所有接受单元 `|Q - ΣQ子|` 之和，不做 Richardson 缩小） |
| `converged` | 是否真正达到请求精度；预算（`max_depth`/`max_evaluations`）耗尽时为 `False`，绝不静默当作成功 |
| `evaluations` | 实际连续场求值次数 |
| `depth` | 实际到达的最深细分层数 |
| `region` | 精确积分矩形 `(x0, x1, y0, y1)` |

自适应积分约定（防过早停止）：

- 每个单元为 3x3 张量 Gauss–Legendre；积分区域按四叉树逐步细分。
- 所有单元**至少细分 `min_depth` 层**（默认 2）后才允许判收敛，亮暗快速
  变化的区域不会因少量样本碰巧相同而提前停止。
- 单元接受判据：`|Q单元 - ΣQ四子单元|` 不超过按面积比例分配的全局容差
  `T = max(atol, rtol * Q全区域)`。
- 不同层级的 Gauss 节点永不重合（节点为半宽的无理数倍），父/子估计使用
  真正独立的采样点集。

确定性孔径生成（均为纯函数，输出 `float` 的 0.0/1.0）：

```python
circular_aperture(n, radius, dx)
    # 含边界：hypot((i-n//2)dx, (j-n//2)dx) <= radius 的像元为 1
rectangular_aperture(n, width, height, dx)
    # 沿轴0宽 width、沿轴1高 height，含边界；半宽/半高闭区间判定
```

## 错误语义（非法输入不静默纠正）

- `TypeError`：物理量不是实数（`bool` 按数值传入同样拒绝）；`n` / `pad` 不是
  严格 `int`（**`bool` 即使是 `int` 子类也拒绝**）；孔径元素不是数值；孔径不可
  迭代或行不可迭代。
- `ValueError`：任何实数 / 复数元素为 `NaN` 或 `±Infinity`；`dx/wavelength/z ≤ 0`；
  `radius/width/height < 0`；空阵、非矩形、非方阵；N<1、N>32、M<N、M>64。
- 积分接口额外约束：区域边界须为有限实数且 `x0 < x1`、`y0 < y1`；
  `rtol > 0`、`atol >= 0`；`min_depth/max_depth/max_evaluations` 为严格 `int`
  （`bool` 拒绝）且 `min_depth <= max_depth`；`pixel_energy` 的 `i, j` 为
  `[0, M)` 内严格 `int`。
- 错误信息包含**可定位参数名**（如 `wavelength` 或 `aperture[i][j]`）。
- 计算为纯函数，无批次 / 有状态写操作，因此不存在部分变更。自适应积分的
  细分预算耗尽**不是**静默成功：以 `converged=False` 显式报告（`energy`
  仍为当前最优估计），与正常收敛严格区分。

## 浮点误差口径（不以宽容差掩盖边界错误）

- DFT 与直接定义（逐项复指数求和，见测试内独立参考）逐 bin 比较：
  长度 1、2、3、5、7、8、12、16、31、32、48、64 上相对误差 < **1e-11**（按最大
  幅值归一）。
- Parseval 相对功率误差：N=1…32 实测量级 ≤ **1e-13**（demo 中为 −1.1e-14 量级），
  测试判据为 **1e-12**（这是纯累加舍入，不包含任何近似模型误差）。
- 矩形孔远场与**离散解析式**（有限等比级数闭式解，非 DFT 计算）逐像元比较
  < 1e-12（按全场最大幅值归一）；单像元常强度用精确相等 / 1e-15 判据。
- 坐标为整数倍 `du`，用 `1e-20` 绝对判据核对；DC 像元坐标严格为 `0.0`。
- 像素能量：常强度区域积分与解析值 `scale²·面积` 比较 < 1e-12；散斑区域与
  测试文件内**独立**固定网格 5 点 Gauss 积分参考比较 < 1e-7 相对；相邻 2x2
  像素能量之和与同一大区域单次积分互相核对 < 1e-7 相对（demo 实测约 1e-11）。

## 测试的独立性

`tests/test_fourier_diffraction.py` 内的期望值全部来自本文件自带的独立实现：
暴力 O(n²) DFT、矩形孔有限等比级数闭式解、手工 Parseval 求和。被测核心从不参与
生成期望值；正反 DFT 往返仅作为补充一致性测试。

覆盖：自研 DFT 定义一致性、单像元常强度与线性相位、矩形孔离散解析式、圆孔确定性
与对称性、Parseval 独立核功率、零填充功率不变与间距更新、相位斜坡导致的整数像元
平移（移位定理）、奇偶网格 fftshift 坐标约定、全部错误语义。

积分测试的独立参考（`reference_field` / `reference_integral`，均在测试文件内
重写）：连续场用逐项 `cmath.exp` 直接求和（不用被测的相位幂次递推），积分用
固定网格 5 点 Gauss–Legendre（与被测的 3 点自适应规则阶数、节点、细分策略
均不同）。另含：连续场与 DFT bin 的一致性、`min_depth` 强制细分（常 integrand
也必须达到指定深度）、预算耗尽时 `converged=False` 且同区域合理容差可收敛、
相邻像素可加性交叉核对、像素区域坐标精确性。

## 设计取舍

- Fraunhofer（远场）模型是本库唯一物理模型；不实现 Fresnel / 角谱传播。
- 居中嵌入与 fftshift 采用统一模运算映射（非负、不创建负索引），对奇偶 N、M 一致。
- DFT 输入规模很小（≤64），优先正确性与可读性；混合基 + 素数回退使 64 长度变换
  仍是毫秒级，不为此引入第三方库。
- 接口只接受 Python 原生嵌套序列，返回原生 list，无 numpy 依赖。
- 像素能量走连续场直接求和（每点 O(N²)），自适应细分把求值集中在亮暗变化
  快的区域；`max_evaluations` 预算保证最坏情况耗时可控，耗尽时显式报告
  不收敛而不是返回无保证的数值。

## 已知限制

- 仅实现本题契约的数学子集：标量传播、均匀采样、单次传播；**不宣称**与任何行业
  标准光学软件兼容。
- 不做近场/倾斜因子/孔径内振幅包络以外的物理效应；`U` 的全局常数相位
  Fraunhofer 相位项按题面公式省略。
- 64 点素数回退路径只在非 2/3 因子长度出现（如 M 为素数，最大为 61），仍为
  O(n²)，在本题规模下耗时以毫秒计。
- 坐标与强度单位假定全部输入以米给出，库内不做单位换算。
- 连续场把孔径像元 `(m, n)` 置于物理坐标 `((m-n//2)dx, (n-n//2)dx)`（点采样
  模型，与 DFT 约定一致）；不包含孔径像元本身的面积平均效应。
