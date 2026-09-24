# slang-KANs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Slang](https://img.shields.io/badge/Slang-2024+-blueviolet.svg)](https://shader-slang.com/)
[![Backends](https://img.shields.io/badge/Backends-Vulkan%20%7C%20Metal%20%7C%20CUDA%20%7C%20CPU-orange.svg)](https://shader-slang.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**slang-KANs**: библиотека сетей Колмогорова-Арнольда (**Kolmogorov-Arnold Networks, KAN**) на языке шейдеров **Slang** с поддержкой **Vulkan, Metal, CUDA и CPU**.

Библиотека обеспечивает полную эквивалентность API с `mlx-KANs` и `metal-KANs`, исполняясь через JIT-компилятор Slang (`slangpy`) с прямым аппаратным доступом к разделяемой памяти тредгрупп (SRAM).

---

## Ключевые возможности

- **10 архитектур KAN**:
  - `FastKAN`: Гауссовы RBF-базисы с общими центрами.
  - `ReLUKAN`: Кусочно-линейные палаточные функции (0 трансцендентных операций).
  - `ChebyKAN`: Полиномы Чебышёва с рекуррентностью в регистрах потоков.
  - `WavKAN`: Непрерывные вейвлеты (Mexican Hat, Morlet, DOG).
  - `FourierKAN`: Гармонические тригонометрические ряды ($\cos, \sin$).
  - `JacobiKAN`: Обобщенные ортогональные полиномы Якоби $(\alpha, \beta)$.
  - `RationalKAN`: Рациональные дроби Паде-Чебышёва ($P/Q$).
  - `KAN`: Классический алгоритм Кокса-де Боора для B-сплайнов.
  - `MultKAN`: KAN 2.0 с явными узлами умножения ($u \cdot v$).
  - `LowRankKAN`: Факторизованная проекция со сжатием параметров до 10 раз.
- **16x16 Shared-Memory Tiled GEMM (v0.2.0)**: Кернел матричного умножения на базе разделяемой памяти тредгрупп (`groupshared float sA[16][16], sB[16][16]`), устраняющий до 85% избыточных обращений к VRAM.
- **Субмиллисекундное время отклика**: На малых батчах ($B=128$) `slang-KANs` опережает Metal и MLX до **3.05x** (0.132 ms).
- **Кросс-платформенность**: Автоматическое определение бэкенда (Metal на Apple Silicon, Vulkan/CUDA на Linux/Windows, CPU fallback).
- **Квантование INT8 и INT4**: Блочно-аффинное сжатие весов (`to_int8`, `to_int4`).
- **Символьная регрессия и экспорт в C99**: Преобразование в аналитические формулы (`to_symbolic`) и автономный заголовочный файл C99.
- **Встроенные оптимизаторы**: Полный набор кроссплатформенных оптимизаторов на чистом NumPy: `AdamW`, `Muon` (с 5-м порядком полиномов Ньютона-Шульца для матриц весов), `Lion`, `RMSprop`, `Adam` и `SGD`.

---

## Бенчмарки производительности

### Сравнение 3 библиотек на GPU: MLX vs metal-KANs (v0.3.0) vs slang-KANs (v0.2.0)

Тестирование на **Apple Silicon GPU**, слой `64 -> 64`:

| Архитектура | Батч | MLX (ms) | metal-KANs (Pure Metal AMX) | slang-KANs (Shared-Memory GEMM) | Победитель (Ускорение) |
|---|---|---|---|---|---|
| **ChebyKAN** | 128 | 0.321 ms | 0.350 ms | **0.166 ms** | **slang-KANs (1.94x)** |
| | 1024 | 1.269 ms | 0.509 ms | **0.369 ms** | **slang-KANs (1.38x)** |
| | 4096 | 1.383 ms | **0.653 ms** | 0.892 ms | **metal-KANs (1.37x)** |
| **BSplineKAN** | 128 | 0.359 ms | 0.290 ms | **0.132 ms** | **slang-KANs (2.20x)** |
| | 1024 | 0.777 ms | **0.457 ms** | 0.545 ms | **metal-KANs (1.19x)** |
| | 4096 | 2.092 ms | **0.902 ms** | 1.402 ms | **metal-KANs (1.56x)** |
| **FastKAN** | 128 | **0.267 ms** | 0.297 ms | 0.696 ms | **MLX (1.11x)** |
| | 1024 | 0.446 ms | **0.399 ms** | 1.463 ms | **metal-KANs (1.12x)** |
| | 4096 | 1.445 ms | **1.177 ms** | 4.739 ms | **metal-KANs (1.23x)** |
| **WavKAN** | 128 | **0.335 ms** | 0.411 ms | 0.833 ms | **MLX (1.23x)** |
| | 1024 | 1.528 ms | **0.567 ms** | 1.835 ms | **metal-KANs (2.69x)** |
| | 4096 | 1.222 ms | **0.984 ms** | 3.762 ms | **metal-KANs (1.24x)** |
| **ReLUKAN** | 128 | **0.309 ms** | 0.378 ms | 0.861 ms | **MLX (1.22x)** |
| | 1024 | **0.513 ms** | 0.557 ms | 1.840 ms | **MLX (1.08x)** |
| | 4096 | **1.093 ms** | 1.135 ms | 3.250 ms | **MLX (1.04x)** |
| **FourierKAN** | 128 | **0.321 ms** | 0.381 ms | 0.896 ms | **MLX (1.19x)** |
| | 1024 | 0.872 ms | **0.641 ms** | 1.969 ms | **metal-KANs (1.36x)** |
| | 4096 | 2.219 ms | **0.928 ms** | 2.565 ms | **metal-KANs (2.39x)** |
| **JacobiKAN** | 128 | 0.292 ms | 0.298 ms | **0.144 ms** | **slang-KANs (2.03x)** |
| | 1024 | 0.578 ms | 0.465 ms | **0.359 ms** | **slang-KANs (1.30x)** |
| | 4096 | 1.521 ms | **0.574 ms** | 0.727 ms | **metal-KANs (1.27x)** |
| **RationalKAN**| 128 | 0.679 ms | 0.403 ms | **0.132 ms** | **slang-KANs (3.05x)** |
| | 1024 | 4.070 ms | 0.850 ms | **0.652 ms** | **slang-KANs (1.30x)** |
| | 4096 | 18.252 ms| 2.453 ms | **2.393 ms** | **slang-KANs (1.02x)** |
| **MultKAN** | 128 | **0.320 ms** | 0.509 ms | 0.803 ms | **MLX (1.59x)** |
| | 1024 | **0.428 ms** | 0.707 ms | 1.561 ms | **MLX (1.65x)** |
| | 4096 | **1.529 ms** | 1.853 ms | 6.638 ms | **MLX (1.21x)** |
| **LowRankKAN** | 128 | 0.344 ms | 0.454 ms | **0.301 ms** | **slang-KANs (1.14x)** |
| | 1024 | **0.608 ms** | 0.829 ms | 3.306 ms | **MLX (1.36x)** |
| | 4096 | **1.442 ms** | 2.257 ms | 11.785 ms| **MLX (1.57x)** |

---

## Установка

```bash
git clone https://github.com/Shuril/slang-kans.git
cd slang-kans
pip install -e .
```

---

## Пример использования

```python
import numpy as np
import slang_kans

# Инициализация слоя FastKAN
layer = slang_kans.FastKAN(layers_hidden=[4, 32, 2], num_centers=8)

# Прямой проход на GPU
x = np.random.uniform(-1.0, 1.0, (128, 4)).astype(np.float32)
y = layer(x)
print("Выходной тензор:", y.shape)  # (128, 2)

# Квантование модели в INT8
q_layer = slang_kans.to_int8(layer)

# Экспорт формулы и C-кода
cheby = slang_kans.ChebyKANLinear(2, 1, degree=3)
sym = slang_kans.to_symbolic(cheby)
print("Формула:", sym.formula())
c_code = sym.to_c_code(func_name="evaluate_kan")

# Встроенные оптимизаторы (AdamW, Muon, Lion, RMSprop, SGD)
from slang_kans import AdamW, Muon, Lion
opt = Muon([(cheby.weights, grad_weights)], lr=0.02, momentum=0.95)
opt.step()
```

---

## Лицензия

Распространяется под лицензией [MIT](LICENSE).
