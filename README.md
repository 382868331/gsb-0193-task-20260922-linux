# fourier_diffraction — 复杂孔径的傅里叶衍射采样

离线计算光学用的可复用小库：对采样的复振幅孔径按**固定 Fraunhofer（远场）模型**做
二维 DFT 传播，返回远场强度、物理坐标、总功率误差与 DC 中心索引。二维 DFT 在本库内
自行实现（迭代 radix-2 Cooley–Tukey，非二幂长度回退为直接求和），**不调用任何 FFT
包或第三方数值库**。

## 环境与依赖

- Python **3.14.7**（3.10+ 亦可运行，目标版本为 3.14.7），Linux 原生。
- 仅标准库（`cmath`、`math`、`dataclasses`、`unittest`、`random` 等）。
- 安装 Python 后无需 `pip install` 任何东西；核心、测试、演示全部离线，
  不访问网络、账号、数据库、中间件、云 API 或 Docker。

## 运行命令

```bash
python -m unittest discover -s tests -v   # 必要测试
python demo.py                             # 固定输入演示（约 1 秒，远小于 8 秒）
```

两条命令都在仓库根目录执行。

## 物理模型（本库固定的数学子集）

```
U[p,q] = dx² / (wavelength · z) · DFT2(a_padded)[p,q]
dx_out = wavelength · z / (M · dx)
```

- 输入孔径 `a` 为 `N×N` 复振幅网格，`1 ≤ N ≤ 32`；`dx`、`wavelength`、`z`
  为正有限实数，单位自洽即可。
- DFT 为未归一化约定：`X[k] = Σ_n x[n]·exp(−2πi·k·n/N)`，原点在角点（DC 在
  索引 0）。
- `pad_to=M` 可选，`N ≤ M ≤ 64`，将孔径置于 `M×M` 零画布角点；零填充只是
  细化观测面采样，**`dx_out` 始终按 M 重新计算并如实返回**，不复用未填充间距。
  标度 `dx²/(λz)` 与 M 无关。
- 返回结果经过标准 `fftshift`（循环平移 `(M//2, M//2)`）。

### 偶数网格 fftshift 约定（明确口径）

- 平移后输出索引 `i` 对应原始频率 `k = i − M//2 (mod M)`。
- **DC 恒在索引 `(M//2, M//2)`**，其物理坐标为 `(x,y)=(0,0)`（输出网格
  在原点有采样）。
- 偶数 M 的整数频率网格为 `−M/2 … M/2−1`：包含负 Nyquist 边
  `−M/2·dx_out`，不包含重复的正边 `+M/2·dx_out`（不对称只在边缘，不在中心）；
  奇数 M 为对称网格 `−(M−1)/2 … (M−1)/2`。
- 物理坐标：`x[j] = (j − M//2)·dx_out`（列），`y[i] = (i − M//2)·dx_out`（行）。
- 注意输入孔径的零居中采样是另一个格点约定：偶数 N 时
  `x[i]=(i−(N−1)/2)·dx`，四个像元环绕原点、原点上无采样；它在角点嵌入的
  DFT 中表现为整体半采样线性相位，对本库返回的强度/功率没有影响。
  相位斜坡测试据此验证 DFT 平移定理。

## 公开接口

```python
from fourier_diffraction import (
    fraunhofer_propagate, FraunhoferResult,
    circular_aperture, rectangular_aperture, grid_coordinates,
    DiffractionError, MAX_APERTURE_N, MAX_PADDED_M,
)
```

- `fraunhofer_propagate(aperture, dx, wavelength, z, *, pad_to=None) -> FraunhoferResult`
  - `aperture`：`N×N` 的 `list[list]`，元素为 int/float/complex（bool 拒绝）。
  - `FraunhoferResult`（不可变 dataclass）字段：
    `intensity`（`M×M`，`|U|²`）、`field`（`M×M` 复场）、`x`、`y`（物理坐标，
    长度 M）、`pixel_size`（=`dx_out`）、`center`（=`(M//2,M//2)`）、
    `input_power`（`dx²·Σ|a|²`）、`output_power`（`dx_out²·Σ|U|²`）、
    `relative_power_error`（`(P_out−P_in)/P_in`；全零孔径时为 0）、
    `input_size=N`、`padded_size=M`；方法 `peak() -> (row, col, intensity)`。
- `circular_aperture(n, dx, radius)`：`sqrt(x²+y²) <= radius` 边界含等号，
  孔内透过率 `1+0j`。
- `rectangular_aperture(n, dx, width, height)`：`|x| ≤ width/2` 且
  `|y| ≤ height/2`，边界含等号。
- `grid_coordinates(n, dx)`：输入面像元物理坐标 `x[i] = (i−(n−1)/2)·dx`，
  偶数网格时原点位于四像元之间。

## 输入上限与错误语义

- 孔径 `N ≤ 32`；零填充 `M ≤ 64`；超限、非方阵、空输入、行非序列、元素非数、
  `pad_to` 为浮点/bool/越界等，一律抛 `DiffractionError`。
- `DiffractionError` 是 `ValueError` 子类，`.parameter` 给出可定位的名字
  （如 `"wavelength"`、`"pad_to"`、`"aperture[1][4]"`），错误信息含实际值；
  **不静默纠正、不夹断、不隐式取整**。
- 数值标量与复像元拒绝 **NaN / ±Infinity**；所有整数参数（`n`、`pad_to` 及
  物理孔径生成器的尺寸参数位）**拒绝 bool**；`dx/wavelength/z/radius/width/height`
  必须严格为正。
- 库内无可变批处理/有状态对象：任何失败只抛异常，不产生部分结果；不存在
  “搜索预算耗尽”这一类操作——本库是有限数学子集上的直接计算，无此概念，
  亦不宣称任何行业标准（Fresnel/Fresnel–Kirchhoff、像差、倾斜入射等）兼容性。

## 测试与独立参考（测试如何核对正确性）

- `tests/test_dft.py`：自实现 DFT 对长度 1–16、31/32/33/48/63/64 与二维
  1–16 尺寸，逐点比对**独立直接求和参考**；fftshift 用手工构造谱核对。
- `tests/test_propagator.py`：
  - 标度/网格/坐标对独立 2-D 朴素 DFT 逐点核对；偶数网格坐标显式断言 `−4…3`；
  - **Parseval 独立核对功率**：参考侧用逐行朴素 DFT 求 `Σ|DFT|²`，
    并核对解析公式 `P_out = dx_out²·(dx²/(λz))²·M²·开孔数`；
  - **单像元常强度**（全网格 `|DFT|²` 恒定）；
  - **矩形孔离散解析式参考**：几何级数闭式
    `e^(−iπk(lo+hi)/M)·sin(πkL/M)/sin(πk/M)`，含多种零填充尺寸；
  - **零填充总功率不变**（M=10/12/16/32/64）、间距更新；
  - **相位斜坡位移**：按 DFT 平移定理逐点移动强度图样，并与闭式相位核对；
  - 往返一致性（零填充在重合频率采样一致、DC 一致）仅作为补充测试。
- 独立参考位于 `tests/_reference.py`，**不 import 被测核心**来生成期望值。

### 浮点误差口径（README 明确，不用宽容差掩盖边界错误）

- 逐点场/强度参考比对：相对容差 `1e-10`、绝对容差 `1e-9 ~ 1e-8`
  （M≤64、O(log N) 层蝶形/直接求和的双精度舍入量级；数值放大处按相对容差）。
- 总功率相对误差（`relative_power_error` 与各断言）：**`|·| < 1e-10`**。
- 坐标/间距标量：`1e-14 ~ 1e-15`（本质是一次乘除）。
- 单像元强度全网格散布：`< 1e-12`。
- 这些容差是双精度浮点噪声上限，不放宽任何物理/索引断言；索引、形状、
  开孔集合、峰值位置全部按精确整数断言。

## 设计取舍

- 角点嵌入零画布（而不是中心化嵌入）：物理孔径居中仅改变线性相位，强度、
  功率、所有验收量不变；实现与解析式参考更直接。
- radix-2 蝶式 + 旋转因子缓存保证 64×64 纯 Python 亚秒级；非二幂长度
  （契约内可能出现，如 N=31 不填充）走直接 DFT，不拒绝合法输入。
- 强度由复场直接算 `re²+im²`（不经 `abs` 的额外开方/平方），输出功率用
  `math.fsum` 求和再乘 `dx_out²`，减小求和顺序误差。
- 孔径生成器只产生确定的二值实孔（确定性、可重复）；任意复振幅由调用方
  自行构造后传入。

## 已知限制

- 仅固定 Fraunhofer 模型与上述标度；不含 Fresnel 传播、倾斜因子、瞳函数
  像差、入射场包络或物理单位换算。
- 仅方阵、N≤32、M≤64；非二幂长度使用 O(N²) 直接 DFT（规模上限内仍很快，
  但不是为更大网格设计的）。
- 偶数输出网格取负 Nyquist 边、舍正边（标准 fftshift 约定，见上），
  不做半像素偏移或对称化重采样；输入孔径偶数 N 时原点上无采样、四像元环绕，
  这是零居中孔径约定，与输出面 DC 落在 `(0,0)` 并不矛盾。
- 不做绘图（无第三方依赖）；`demo.py` 打印归一化中心剖面的数值行展示主峰。
