# 复杂孔径的傅里叶衍射采样（Fraunhofer + 有限距离 Fresnel）

纯 Python 标准库实现的可复用离线模块：对 ≤32×32 的复振幅孔径做二维离散傅里叶
变换得到远场（Fraunhofer），或用固定的 Fresnel 传递函数离散核做**有限距离**传播；
返回复场、强度、物理坐标、总功率误差与中心索引；支持确定性圆孔 / 矩形孔
生成与居中零填充（≤64×64）。**不使用任何 FFT 包或第三方数值库**，DFT 为自行实现
的混合基 Cooley–Tukey（素数长度走 O(n²) 直接求和回退），Fresnel 与 Fraunhofer
共用同一变换核心。

## 运行环境

- Python **3.14.7**（仅使用标准库 `cmath` / `math` / `dataclasses`）
- Linux 原生运行，安装 Python 后**无需安装任何其他依赖**，无需网络、账号、密钥、
  数据库、中间件、云 API 或 Docker
- 核心算法、测试、演示全部离线、确定性

## venv 准备与准确命令

```bash
python3 -m venv .venv            # 可选；仅标准库，venv 内无需 pip install
source .venv/bin/activate
python -m unittest discover -s tests -v   # 必要测试（35 个用例）
python demo_iteration.py                  # Fresnel 有限距离演示，现场计算，远小于 90 秒
python demo.py                            # Fraunhofer 远场演示，约 0.1 秒（上限 8 秒）
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

### Fresnel 有限距离传播（固定离散核）

- 同一居中 N×N 输入契约。离散模型为 **Fresnel 传递函数**（作用在 DFT 频谱上）：

  ```
  U      = IDFT[ DFT[A] · H ]                 # 前向 DFT 不归一，逆变换除 N²
  H[k,l] = exp(−i·π·λ·z·(fx_k² + fy_l²))      # 可分离：H[k,l] = h[k]·h[l]
  fx_k   = ((k + N//2) mod N − N//2) / (N·dx) # 居中频率 bin（周/米）
  ```

  复振幅归一化：`|H| = 1` 的幺模传递函数 + 上述 DFT 归一对，载波相位
  `exp(i·2π·z/λ)` 与 Fraunhofer 入口一样按题面省略。
- **输出采样窗与输入相同**：`du = dx`，坐标 `x[p] = (p − N//2)·dx`，
  `center = (N//2, N//2)`，无零填充参数。
- `z` 为**有符号**有限实数：负值表示反向传播（`H(−z) = conj(H(z))`）；
  `z = 0` 直接返回输入复场的拷贝，**不做除以零**。
- 能量归一：`|H| = 1` 使离散 Parseval 严格成立（仅舍入误差），
  `dx²·Σ|U|² = dx²·Σ|A|²`；`power_error` 口径与 Fraunhofer 相同，
  `z = 0` 或零输入时为 `0.0`。
- 计算复用核心一维变换：行/列 `_dft1` 正变换 → 逐点乘可分离核 → 行/列逆变换，
  不引入任何 FFT 库或优化器。
- **混叠适用范围**：传递函数模型把孔径频谱按 N×N 采样窗周期化。仅当孔径频谱
  在最高 bin（`|fx|, |fy| ≤ 1/(2·dx)`）以外能量可忽略、且最高频处相位增量
  `π·λ·|z|·f_max²` 仍被采样分辨时，离散结果才逼近连续 Fresnel 衍射；
  强聚焦 / 过大 `|z|` 使孔径高频含量显著时会混叠。**本模块不宣称**离散结果
  是任意采样条件下连续光场的精确解。

## 接口

```python
from fourier_diffraction import fraunhofer, circular_aperture, rectangular_aperture

r = fraunhofer(aperture, dx, wavelength, z, pad=None) -> FraunhoferResult
```

Fresnel 有限距离入口（同一模块、同一孔径契约）：

```python
from fourier_diffraction import fresnel

r = fresnel(aperture, dx, wavelength, z) -> FresnelResult
```

- `z`：**有符号**有限实数（米）；`z = 0` 返回输入拷贝，负值反向传播。
- `FresnelResult` 字段与 `FraunhoferResult` 同名同义（`intensity` / `field` /
  `x` / `y` / `du` / `center` / `input_power` / `output_power` /
  `power_error` / `n`），只是 `du` 恒等于输入 `dx`、无 `m`，并多一个
  记录实际传播距离的 `z` 字段。

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

## 错误语义（非法输入不静默纠正）

- `TypeError`：物理量不是实数（`bool` 按数值传入同样拒绝）；`n` / `pad` 不是
  严格 `int`（**`bool` 即使是 `int` 子类也拒绝**）；孔径元素不是数值；孔径不可
  迭代或行不可迭代。
- `ValueError`：任何实数 / 复数元素为 `NaN` 或 `±Infinity`；`dx/wavelength/z ≤ 0`
  （`z ≤ 0` 仅针对 Fraunhofer；Fresnel 的 `z` 符号自由、允许 0）；
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
- Fresnel 复场（**含相位**，按复数比较而非只比强度）与测试内独立的 O(N⁴) 直接
  双重求和参考逐像元比较 < **1e-11**（按最大幅值归一，N=8，正 / 负距离、单点
  孔径与一般复孔径）；平面波不变性判据 1e-15；相位斜坡获得精确核相位判据
  1e-12；正反向往返误差 < 1e-12；Parseval `power_error` 判据 1e-12。

## 测试的独立性

`tests/test_fourier_diffraction.py` 内的期望值全部来自本文件自带的独立实现：
暴力 O(n²) DFT、矩形孔有限等比级数闭式解、手工 Parseval 求和。被测核心从不参与
生成期望值；正反 DFT 往返仅作为补充一致性测试。

覆盖：自研 DFT 定义一致性、单像元常强度与线性相位、矩形孔离散解析式、圆孔确定性
与对称性、Parseval 独立核功率、零填充功率不变与间距更新、相位斜坡导致的整数像元
平移（移位定理）、奇偶网格 fftshift 坐标约定、全部错误语义。

`tests/test_fresnel.py` 的期望值同样来自文件内独立实现：暴力 O(N⁴) 二维直接复数
求和（前向 DFT、逐点乘传递函数、逆直接求和），不调用被测核心。覆盖：一般复孔径 /
单点孔径 / 负距离的复场（含相位）一致性、`z=0` 原样返回输入、平面波不变、相位
斜坡获得精确核相位、正反向传播往返误差（补充一致性）、输出采样窗与坐标、
Parseval 能量守恒、零输入功率口径、全部 Fresnel 错误语义（`z` 为 bool / 非实数 /
NaN / ±inf，共享孔径校验，错误信息含参数名）。

## 设计取舍

- 物理模型固定为两个：Fraunhofer（远场）与 Fresnel 传递函数（有限距离）；
  不实现角谱 / Rayleigh–Sommerfeld 等其他传播模型，也不引入优化器。
- Fresnel 选传递函数形式而非脉冲响应卷积：核可分离、`|H|=1` 使离散能量严格
  守恒、正反向严格互逆，且天然复用既有行/列一维变换。
- 居中嵌入与 fftshift 采用统一模运算映射（非负、不创建负索引），对奇偶 N、M 一致。
- DFT 输入规模很小（≤64），优先正确性与可读性；混合基 + 素数回退使 64 长度变换
  仍是毫秒级，不为此引入第三方库。
- 接口只接受 Python 原生嵌套序列，返回原生 list，无 numpy 依赖。

## 已知限制

- 仅实现本题契约的数学子集：标量传播、均匀采样、单次传播；**不宣称**与任何行业
  标准光学软件兼容。
- Fresnel 结果为**固定采样窗上的离散模型**：仅在孔径频谱被采样窗充分容纳时逼近
  连续 Fresnel 衍射（混叠适用范围见上文“Fresnel 有限距离传播”一节），不宣称是
  任意采样条件下连续光场的精确解。
- 不做近场/倾斜因子/孔径内振幅包络以外的物理效应；`U` 的全局常数相位
  Fraunhofer 相位项按题面公式省略。
- 64 点素数回退路径只在非 2/3 因子长度出现（如 M 为素数，最大为 61），仍为
  O(n²)，在本题规模下耗时以毫秒计。
- 坐标与强度单位假定全部输入以米给出，库内不做单位换算。
