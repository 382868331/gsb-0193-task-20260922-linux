# 复杂孔径的傅里叶衍射采样（Fraunhofer 远场 + Fresnel 有限距离）

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
python -m unittest discover -s tests -v   # 必要测试（37 个用例，含 Fresnel 增量）
python demo.py                            # Fraunhofer 远场演示，约 0.1 秒（上限 8 秒）
python demo_iteration.py                  # Fresnel 有限距离演示，现场计算 < 90 秒（实测约 0.1 秒）
```

## venv 准备（可选但推荐）

仅依赖 Python 3.14.7 标准库，venv 内**不需要安装任何包**：

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m unittest discover -s tests -v
python demo_iteration.py
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

## Fresnel 有限距离传播（本题增量）

```python
from fourier_diffraction import fresnel

r = fresnel(aperture, dx, wavelength, z) -> FresnelResult
```

- `aperture`：与 `fraunhofer` 相同的 N×N 居中复振幅（1 ≤ N ≤ 32）。
- `dx, wavelength`：有限正实数（米）。**输出采样窗与输入完全相同**：
  N×N 网格、像元间距 `dx`，坐标 `x[p] = (p − N//2)·dx`。
- `z`：有限实数（米），**可正可负**（负值为反向传播，核取共轭符号）；
  `z = 0` 是退化距离，**直接返回输入的副本**，不做除零。

固定的 Fresnel 离散核（可分离卷积，同一采样窗进出）：

```
U[p,q] = dx² / (i·λ·z) · Σ_{t,u} A[t,u]
           · exp(iπ·(((p−t)dx)² + ((q−u)dx)²) / (λz))
```

- 复振幅归一化：保留随 z 变化的 `dx²/(iλz)`；全局常数相位 `exp(i·k·z)`
  省略（不影响强度，与 Fraunhofer 入口的约定一致）。
- 实现：核可分离为两个一维 chirp，每行/每列各做一次一维卷积；卷积通过
  **复用模块自带的一维 DFT**（2N 点零填充使圆周卷积等于截断的线性卷积）
  完成，不调用任何 FFT 库，也不引入优化器。
- 能量归一口径：`input_power = dx²·Σ|A|²`，`output_power = dx²·Σ|U|²`，
  `power_error` 为有符号相对差。与 Fraunhofer 的 Parseval 恒等不同，这里
  **不是纯舍入**：衍射能量散出有限窗口会被真实截断，`power_error` 同时
  报告窗口截断（demo 中约 +2e-2 量级）。
- `FresnelResult` 字段：`intensity / field / x / y / dx / center /
  input_power / output_power / power_error / n / z`，frozen dataclass。

### 混叠与适用范围（不宣称连续光场精确解）

输出是**上述离散和的精确求值**（对 8×8 暴力二维复数求和偏差 ~1e-15），
但只在采样充分时才是连续 Fresnel 光场的近似：

- chirp 无混叠要求 `λ·|z| / (2·dx²)` 不小于实际参与卷积的最大偏移（像元数）；
  `z` 过小时核在采样点间剧烈振荡，离散模型偏离连续物理（demo 的功率误差
  在 z→0 方向会急剧增大，这是采样限制而非实现缺陷）。
- 衍射扩展 `λ·|z|/dx` 应明显小于窗口 `N·dx`，否则能量散出窗口被截断，
  正反向往返不能精确还原（32×32、中心 7×7 孔径、z=6 mm 时往返误差约
  1e-1 量级，测试以显式上界 0.15 固化该口径）。
- 因此本入口不宣称在任意采样条件下给出连续光场的精确解；它精确实现的是
  题内固定的离散模型。

## 错误语义（非法输入不静默纠正）

- `TypeError`：物理量不是实数（`bool` 按数值传入同样拒绝）；`n` / `pad` 不是
   严格 `int`（**`bool` 即使是 `int` 子类也拒绝**）；孔径元素不是数值；孔径不可
   迭代或行不可迭代。
- `ValueError`：任何实数 / 复数元素为 `NaN` 或 `±Infinity`；`dx/wavelength/z ≤ 0`
  （Fresnel 入口中 `z` 允许任意有限实数，含 0 与负值）；
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

覆盖：自研 DFT 定义一致性、单像元常强度与线性相位、矩形孔离散解析式、圆孔确定性
与对称性、Parseval 独立核功率、零填充功率不变与间距更新、相位斜坡导致的整数像元
平移（移位定理）、奇偶网格 fftshift 坐标约定、全部错误语义。

`tests/test_fresnel.py` 的期望值同样来自文件内独立实现：暴力 O(N^4) 二维复数求和
（相位参与比较，不只看强度）、单点孔径闭式 chirp。覆盖：正/负距离与直接和逐像元
比较、单点孔径闭式解、平面相位（线性相位斜坡）复场与质心位移、z=0 原样返回且不
改输入、正反向传播往返误差显式上界、输出窗与输入窗一致、功率字段独立核对、零输入
功率口径、全部异常路径（z 的类型/NaN/±Inf、dx/wavelength 非法、孔径复用基线校验、
错误信息含参数名）。

## 设计取舍

- 物理模型为 Fraunhofer（远场）与 Fresnel 卷积型有限距离传播；不实现角谱方法。
- 居中嵌入与 fftshift 采用统一模运算映射（非负、不创建负索引），对奇偶 N、M 一致。
- DFT 输入规模很小（≤64），优先正确性与可读性；混合基 + 素数回退使 64 长度变换
  仍是毫秒级，不为此引入第三方库。
- 接口只接受 Python 原生嵌套序列，返回原生 list，无 numpy 依赖。

## 已知限制

- 仅实现本题契约的数学子集：标量传播、均匀采样、单次传播；**不宣称**与任何行业
  标准光学软件兼容。
- Fresnel 入口精确实现的是题内固定离散核；仅在 chirp 采样充分且衍射扩展不超出
  采样窗时近似连续光场，**不宣称**任意采样条件下的连续场精确解（详见上文
  “混叠与适用范围”）。
- 不做近场/倾斜因子/孔径内振幅包络以外的物理效应；`U` 的全局常数相位
  Fraunhofer 相位项按题面公式省略。
- 64 点素数回退路径只在非 2/3 因子长度出现（如 M 为素数，最大为 61），仍为
  O(n²)，在本题规模下耗时以毫秒计。
- 坐标与强度单位假定全部输入以米给出，库内不做单位换算。
