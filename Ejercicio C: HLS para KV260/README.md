# Ejercicio C: Softmax con HLS en la AMD Kria KV260

Implementación de **Softmax estable numéricamente** en una FPGA AMD Kria KV260, mediante **Vitis HLS 2024.2**. Se comparan cuatro variantes FP32 y cuatro tamaños de entrada (`N = 10, 100, 500, 1000`), utilizando como referencia cálculos en FP64 (`double`). Se distinguen la latencia del IP obtenida mediante cosimulación C/RTL, las estimaciones de síntesis HLS y las mediciones realizadas en la Kria: transferencias, tiempo de llamada al kernel con XRT, tiempo total amortizado, potencia, energía y error numérico.

**Alcance de este documento:** los datos medidos **FP32** están completos y validados. Se reserva una sección para **FP16**, cuyos resultados serán incorporados por la persona responsable. Se incluyen la **latencia del IP obtenida mediante cosimulación C/RTL**, los **intervalos de iniciación (II)** y los **recursos estimados en síntesis HLS** de las cuatro variantes FP32. Se presentan por separado de los tiempos medidos en el host. Las secciones de **FP16** permanecen pendientes de completar.

---

## 1. Implementación

Softmax normaliza un vector de entrada `x`:

$$
y_i = \frac{\exp(x_i-m)}{\sum_{j=0}^{N-1} \exp(x_j-m)}, \qquad m = \max_j x_j
$$

La resta del máximo evita desbordamiento numérico al calcular la exponencial. La implementación dispone de tres etapas: **máximo**, **exponenciales y suma**, y **normalización**.

### 1.1 Variantes de hardware

| Variante | Características utilizadas | Interfaz visible en XRT |
|---|---|---|
| **V0** | Ejecución secuencial de las tres etapas, sin `PIPELINE` explícito. | `x, y, n` |
| **V1** | Directivas `PIPELINE II=1` en los bucles. | `x, y, n` |
| **V2** | `PIPELINE` y `ARRAY_PARTITION` para la entrada (`cyclic factor=2` en el código proporcionado). | `x_0, x_1, y, n` |
| **V3** | Variante con `DATAFLOW` y optimizaciones de pipeline/partición; el binario proporcionado expone dos puertos de entrada. | `x_0, x_1, y, n` |

**Detalle relevante para V2/V3:** HLS expuso el argumento `x` como dos argumentos AXI de memoria: `x_0` (ID 0) y `x_1` (ID 1); la salida `y` es ID 2 y `n` es ID 3. El host distribuye la entrada de manera cíclica: `x_0` recibe índices pares y `x_1` índices impares. En V0/V1 la interfaz conserva tres argumentos (`x`, `y`, `n`). La distribución fue verificada en la KV260 mediante resultados `PASS` para los cuatro tamaños de entrada.

`PIPELINE`: permite iniciar iteraciones antes de que terminen otras cuando las dependencias lo permiten. `II` (*initiation interval*): ciclos entre inicios de iteraciones. `ARRAY_PARTITION`: divide un arreglo en bancos/puertos. `DATAFLOW`: permite solapamiento entre tareas o etapas compatibles. El beneficio final depende de las dependencias, la arquitectura y el coste de las transferencias.

### 1.2 Funciones del host

El archivo `softmax_host.cpp` realiza lo siguiente:

1. Abre el dispositivo y el `.xclbin` correspondiente mediante **XRT** (*Xilinx Runtime*).
2. Determina la variante por el nombre del `.xclbin`, asigna los buffers a sus `group_id()` y organiza los argumentos del kernel.
3. Genera una entrada determinista en `[-4, 4)` mediante un generador congruencial lineal con **semilla 42**; calcula una referencia `double` (FP64).
4. Ejecuta tres calentamientos antes de medir.
5. Registra los tiempos H2D, `kernel + XRT`, D2H, total y cada llamada individual.
6. Mide potencia mediante el sensor INA260 de la Kria usando un hilo de monitoreo; estima energía como potencia media por tiempo total.
7. Comprueba valores finitos, rango `[0,1]`, suma aproximadamente igual a 1, error absoluto máximo y RMSE.
8. Guarda dos CSV independientes por ejecución (resumen y llamadas individuales).

---

## 2. Reproducción paso a paso

Los comandos siguientes reflejan la organización utilizada en el laboratorio. Existen **dos directorios diferentes**: el proyecto HLS en la laptop (`Ejercicio C: HLS para KV260/`) y el directorio de ejecución de la Kria (`~/lab-4/softmax_FP32/`). Ejecutar cada bloque en el equipo indicado.

### 2.1 Organización de archivos

**Laptop (proyecto HLS):**

```text
Ejercicio C: HLS para KV260/
├── README.md
├── platform/
│   └── kv260_custom.xpfm
├── softmax_FP32_V0/
│   ├── HW/
│   │   ├── softmax_hls.cpp
│   │   ├── softmax_hls.hpp
│   │   ├── testbench_v0.cpp
│   │   ├── Makefile
│   │   ├── softmax_hls/softmax_config.cfg
│   │   └── package.hw/
│   └── reportes/
├── softmax_FP32_V1/
├── softmax_FP32_V2/
├── softmax_FP32_V3/
├── resultados_10/
├── resultados_100/
├── resultados_500/
└── resultados_1000/
```

Las carpetas de las otras versiones (`V1`–`V3`) contienen una estructura equivalente para sus fuentes y reportes. Las carpetas de resultados contienen los CSV recogidos desde la Kria.

**Kria KV260 (ejecución):**

```text
~/lab-4/softmax_FP32/
├── FP32_V0/package.hw/
├── FP32_V1/package.hw/
├── FP32_V2/package.hw/
├── FP32_V3/package.hw/
├── softmax_host.cpp
├── softmax_host
├── resultados_10/
├── resultados_100/
├── resultados_500/
└── resultados_1000/
```

### 2.2 Preparar herramientas en la laptop

```bash
source ~/Xilinx/Vitis/2024.2/settings64.sh
command -v vitis-run
command -v v++
command -v emconfigutil
command -v bootgen
v++ --version
```

Se necesita la plataforma personalizada `platform/kv260_custom.xpfm` y espacio de almacenamiento para la implementación de hardware.

### 2.3 Simulación C, síntesis y cosimulación (laptop)

Desde la raíz del directorio del ejercicio, seleccionar una versión (ejemplo V2):

```bash
cd softmax_FP32_V2/HW
```

**Simulación C (CSIM):**

```bash
vitis-run --mode hls --csim \
    --config ./softmax_hls/softmax_config.cfg \
    --work_dir ./softmax_hls
```

**Síntesis HLS (CSYNTH):**

```bash
v++ -c --mode hls \
    --config ./softmax_hls/softmax_config.cfg \
    --work_dir ./softmax_hls
```

**Cosimulación C/RTL (COSIM):**

```bash
vitis-run --mode hls --cosim \
    --config ./softmax_hls/softmax_config.cfg \
    --work_dir ./softmax_hls
```

El mismo procedimiento se aplica en `softmax_FP32_V0/HW`, `softmax_FP32_V1/HW` y `softmax_FP32_V3/HW`. El archivo `.cfg` debe referenciar correctamente la fuente y el testbench de cada versión. La cosimulación requiere que el componente HLS esté previamente sintetizado.

**Estado de verificación conocido:** V2 pasó CSIM y COSIM con N=10, 100, 500 y 1000, con validación FP32 frente a FP64. Las otras variantes se habían validado durante el desarrollo; para reproducir desde cero deben repetirse las comprobaciones.

### 2.4 Generar el acelerador (laptop)

Desde `HW/` de cada versión:

```bash
make
ls -lh package.hw/
```

El `Makefile` ejecuta la compilación, el enlace y el empaquetado definidos para la plataforma; revisar su contenido mediante `make -n` antes de reconstruir. Los archivos finales esperados son:

| Archivo | Función |
|---|---|
| `softmax_vN_fp32.xclbin` | Metadatos y binario del acelerador utilizados por XRT. |
| `softmax_vN_fp32.bit.bin` | Configuración de la lógica programable. |
| `softmax_vN_fp32.dtbo` | *Device Tree Overlay* para Linux. |
| `shell.json` | Metadatos para `xmutil`. |

Aquí `N` en `vN` representa la variante (`0`, `1`, `2` o `3`), **no** el tamaño del vector. Los cuatro archivos de una variante deben corresponder a la misma compilación.

### 2.5 Copiar los paquetes a la Kria (desde la laptop)

El alias SSH `ich-kria` corresponde a la configuración utilizada en el laboratorio. Desde la raíz del directorio del ejercicio:

```bash
for v in 0 1 2 3; do
    ssh ich-kria "mkdir -p ~/lab-4/softmax_FP32/FP32_V${v}"
    scp -r "softmax_FP32_V${v}/HW/package.hw" \
        "ich-kria:~/lab-4/softmax_FP32/FP32_V${v}/"
done
```

Si no se dispone del alias, configurar la conexión SSH (incluido el salto de red correspondiente) antes de ejecutar los comandos. Copiar los archivos no configura todavía la FPGA.

### 2.6 Registrar y cargar la aplicación FPGA (en la Kria)

```bash
ssh ich-kria
cd ~/lab-4/softmax_FP32
```

Para instalar los archivos de una variante (ejemplo `v=2`):

```bash
v=2
sudo mkdir -p "/lib/firmware/xilinx/softmax_v${v}_fp32"
sudo cp "FP32_V${v}/package.hw/softmax_v${v}_fp32.bit.bin" \
        "FP32_V${v}/package.hw/softmax_v${v}_fp32.dtbo" \
        "FP32_V${v}/package.hw/softmax_v${v}_fp32.xclbin" \
        "FP32_V${v}/package.hw/shell.json" \
        "/lib/firmware/xilinx/softmax_v${v}_fp32/"
```

Comprobar las aplicaciones y cargar la variante deseada (descargar previamente otra aplicación si está activa):

```bash
sudo xmutil listapps
sudo xmutil unloadapp
sudo xmutil loadapp softmax_v2_fp32
sudo xmutil listapps
```

La columna `Active_slot` permite identificar la aplicación activa (en las comprobaciones del laboratorio, `0` indicaba V2 activa y `-1` las inactivas). Para cambiar de variante, repetir `unloadapp` y `loadapp` con el nombre apropiado.

Para verificar que el paquete del host coincide con el instalado:

```bash
sha256sum \
  /lib/firmware/xilinx/softmax_v2_fp32/softmax_v2_fp32.xclbin \
  ~/lab-4/softmax_FP32/FP32_V2/package.hw/softmax_v2_fp32.xclbin
```

En el diagnóstico de V2 se comprobaron además los hashes de `.bit.bin` y `.dtbo`; coincidían con los archivos instalados.

### 2.7 Compilar el host (en la Kria)

Desde `~/lab-4/softmax_FP32/`, usando la versión **actualizada** del archivo `softmax_host.cpp`:

```bash
g++ -O2 -std=c++17 \
    -I/usr/include/xrt \
    softmax_host.cpp -o softmax_host \
    -L/opt/xilinx/xrt/lib \
    -lxrt_core -lxrt_coreutil -pthread
```

`-pthread` se utiliza porque el host mide potencia desde un hilo independiente. **La versión anterior del host no es compatible con V2/V3:** enviaba solo tres argumentos, mientras que los `.xclbin` de estas variantes requieren cuatro. Esto produjo mediciones anteriores que terminaban en `FAIL`; se reemplazaron por corridas validadas.

### 2.8 Ejecutar las mediciones FP32 (en la Kria)

**Protocolo oficial:** `5000` repeticiones del kernel y `3` ejecuciones de calentamiento, para cada N y cada variante. Crear primero las carpetas:

```bash
cd ~/lab-4/softmax_FP32
mkdir -p resultados_{10,100,500,1000}
```

**Ejemplo: V2, N=10.** Con `softmax_v2_fp32` activa:

```bash
cd resultados_10
../softmax_host ../FP32_V2/package.hw/softmax_v2_fp32.xclbin 5000 10 3
```

**Bucle para los cuatro tamaños de una variante**, desde `~/lab-4/softmax_FP32/` (ejemplo V2; cambiar `v` y cargar la aplicación correspondiente antes de ejecutarlo):

```bash
v=2
for n in 10 100 500 1000; do
    (cd "resultados_${n}" && \
      ../softmax_host "../FP32_V${v}/package.hw/softmax_v${v}_fp32.xclbin" 5000 "$n" 3)
done
```

Para repetir las 16 configuraciones, ejecutar el bloque para `v=0,1,2,3`, cambiando la aplicación mediante `xmutil` antes de cada versión. No iniciar una medición si la aplicación activa no coincide con el `.xclbin` utilizado.

**Salidas:** cada proceso crea en su directorio actual dos archivos con identificador temporal:

```text
resultados_softmax_fp32_VN_<id>_resumen.csv
resultados_softmax_fp32_VN_<id>_llamadas.csv
```

`resumen.csv` contiene una fila con las métricas del lote. `llamadas.csv` registra la duración de cada una de las 5000 llamadas. Los calentamientos no se incluyen.

### 2.9 Recuperar los resultados (desde la laptop)

Desde el directorio `Ejercicio C: HLS para KV260/`:

```bash
for n in 10 100 500 1000; do
    scp -r "ich-kria:~/lab-4/softmax_FP32/resultados_${n}" .
done
```

Los nombres de cada archivo contienen un ID temporal; las corridas repetidas no deben eliminarse sin antes identificar cuáles tienen `PASS`. En este documento únicamente se utilizan las corridas oficiales listadas en la sección 9. El patrón remoto terminado en `/.` provocó `unexpected filename: .` en el entorno usado, por lo que no se utiliza.

---

## 3. Protocolo y entorno de medición

| Parámetro | Configuración |
|---|---|
| Plataforma | AMD Kria KV260 (K26 SoM) |
| Sistema Kria | Ubuntu 22.04, arquitectura AArch64 |
| XRT en Kria | 2.13.0 |
| Desarrollo y síntesis | AMD Vitis/Vivado 2024.2 en laptop Ubuntu 22.04 |
| Plataforma Vitis | `kv260_custom.xpfm` |
| Precisión aquí evaluada | FP32 (`float`) |
| Referencia numérica | FP64 (`double`) |
| Tamaños | N=10, 100, 500, 1000 |
| Repeticiones | 5000 por configuración |
| Calentamiento | 3 llamadas, excluidas de las mediciones |
| Entrada | Generador congruencial lineal determinista, semilla 42; valores en [-4, 4) |
| Potencia | Sensor INA260, atributo `power1_input`, lectura en hilo aparte |
| Criterios de validación | Salidas finitas en [0,1]; error máximo ≤ 1e-5; desviación de suma respecto a 1 ≤ 1e-4 |

**Definiciones de los tiempos:**

- **H2D** (*host to device*): duración de una sincronización de entrada al dispositivo **por lote**. Para V2/V3 incluye sincronizar los dos buffers `x_0` y `x_1`.
- **Kernel + XRT:** promedio por ejecución de `softmax(...)` más `run.wait()`, medido desde el host; incorpora el coste de control, lanzamiento y espera, **no equivale a la latencia interna aislada del IP**.
- **D2H** (*device to host*): duración de una sincronización del buffer de salida **por lote**.
- **Tiempo total amortizado:** desde la sincronización de entrada hasta la sincronización de salida, dividido entre 5000 llamadas. **No** significa que se repita H2D y D2H en cada llamada; las transferencias ocurren una sola vez por lote, por lo que su coste se amortiza.
- **Energía por llamada:** `potencia_media × tiempo_total_lote / 5000` (W·µs = µJ). Incluye lo observado por el sensor durante el lote, **sin descontar consumo en reposo**; no representa necesariamente energía exclusiva del IP.

El tiempo de preparación de la entrada y empaquetado en bancos se realiza antes de los cronómetros H2D y total. La verificación numérica se realiza después, sobre **la salida de la última ejecución** del lote. Cada configuración corresponde a una sola corrida de 5000 llamadas; no se calcularon estadísticas entre corridas independientes.

---

## 4. Precisión FP32 frente a FP64

El host calcula la referencia en **FP64** (`double`) y verifica la salida en **FP32** (`float`). Para cada tamaño se comprueba que los valores sean finitos, estén en el intervalo `[0, 1]`, sumen aproximadamente 1 y cumplan el umbral de error absoluto máximo de `1e-5`.

La raíz del error cuadrático medio (**RMSE**) se define como:

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1}(y_i - y_i^{\mathrm{ref}})^2}
$$

Todas las variantes obtuvieron **PASS** para N = 10, 100, 500 y 1000. Los valores de **error máximo** se detallan en la tabla de cada versión, en la sección 5. La validación se aplica a la salida descargada después de la última ejecución del lote, no a cada una de las 5000 llamadas.

---

## 5. Resultados FP32 en la Kria KV260: una tabla por versión

**La latencia del IP obtenida mediante cosimulación C/RTL se presenta por separado en la sección 6.1.**

Se presentan los datos medidos de las **16 corridas oficiales** (5000 llamadas y 3 calentamientos por configuración). Las columnas **H2D** y **D2H** corresponden a una transferencia **por lote**; **Kernel + XRT**, **tiempo total** y **energía** están expresados **por llamada** (total amortizado). Así se evita confundir las transferencias individuales con el tiempo total dividido entre 5000.

- **H2D / D2H:** transferencias entre el host y el dispositivo, en microsegundos (µs).
- **Kernel + XRT:** llamada al kernel y espera `run.wait()`, en µs.
- **Tiempo total:** tiempo completo del lote dividido entre 5000, en µs.
- **J/elem.:** energía estimada por elemento procesado, en julios.

### 5.1 V0 — Secuencial

| N | H2D lote (µs) | Kernel + XRT (µs) | D2H lote (µs) | Tiempo total (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 9.180 | 51.543 | 7.060 | 51.552 | 3.637 | 187.472 | 1.875e-05 | 1.490e-08 |
| 100 | 5.760 | 55.456 | 6.760 | 55.464 | 3.620 | 200.806 | 2.008e-06 | 1.051e-08 |
| 500 | 7.110 | 75.066 | 6.710 | 75.075 | 3.572 | 268.152 | 5.363e-07 | 8.361e-10 |
| 1000 | 4.880 | 100.546 | 6.310 | 100.554 | 3.542 | 356.150 | 3.561e-07 | 2.062e-09 |

**Validación:** PASS en los cuatro tamaños.

### 5.2 V1 — PIPELINE

| N | H2D lote (µs) | Kernel + XRT (µs) | D2H lote (µs) | Tiempo total (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10.080 | 51.591 | 7.200 | 51.599 | 3.647 | 188.167 | 1.882e-05 | 1.490e-08 |
| 100 | 5.160 | 53.006 | 8.080 | 53.015 | 3.636 | 192.784 | 1.928e-06 | 1.051e-08 |
| 500 | 7.670 | 65.603 | 7.220 | 65.612 | 3.595 | 235.904 | 4.718e-07 | 8.361e-10 |
| 1000 | 6.220 | 80.695 | 7.640 | 80.704 | 3.584 | 289.236 | 2.892e-07 | 2.062e-09 |

**Validación:** PASS en los cuatro tamaños.

### 5.3 V2 — PIPELINE + ARRAY_PARTITION

| N | H2D lote (µs) | Kernel + XRT (µs) | D2H lote (µs) | Tiempo total (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 7.660 | 53.069 | 7.440 | 53.073 | 3.691 | 195.911 | 1.959e-05 | 1.490e-08 |
| 100 | 8.270 | 57.357 | 6.530 | 57.362 | 3.704 | 212.493 | 2.125e-06 | 1.051e-08 |
| 500 | 9.360 | 74.047 | 6.360 | 74.052 | 3.796 | 281.075 | 5.621e-07 | 8.361e-10 |
| 1000 | 10.280 | 95.961 | 6.840 | 95.966 | 3.871 | 371.523 | 3.715e-07 | 2.062e-09 |

**Validación:** PASS en los cuatro tamaños.

### 5.4 V3 — DATAFLOW

| N | H2D lote (µs) | Kernel + XRT (µs) | D2H lote (µs) | Tiempo total (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 10.540 | 53.486 | 7.790 | 53.492 | 3.684 | 197.078 | 1.971e-05 | 1.490e-08 |
| 100 | 7.870 | 59.980 | 7.291 | 59.984 | 3.693 | 221.532 | 2.215e-06 | 1.051e-08 |
| 500 | 8.350 | 90.814 | 7.150 | 90.819 | 3.752 | 340.746 | 6.815e-07 | 8.361e-10 |
| 1000 | 8.350 | 127.572 | 6.700 | 127.576 | 3.779 | 482.083 | 4.821e-07 | 2.062e-09 |

**Validación:** PASS en los cuatro tamaños.

Las cifras de energía resultan de multiplicar la potencia media por el tiempo total de ejecución del lote y dividir entre las 5000 llamadas. **No** representan necesariamente energía exclusiva del IP; incluyen el consumo observado por el sensor durante el intervalo medido.

---

## 6. Latencia del IP por cosimulación C/RTL y síntesis HLS (FP32)

La **latencia del IP** se obtuvo mediante la **cosimulación C/RTL con XSim** de Vitis HLS 2024.2. Los registros individuales se encuentran en `softmax_hls.performance.result.transaction.xml` y corresponden a cuatro transacciones por variante. Los testbenches ejecutan una vez cada tamaño de entrada, en el orden **N=10, 100, 500, 1000**; por ello, las transacciones 0 a 3 se asocian a esos valores, respectivamente.

Estos resultados describen el **comportamiento RTL simulado del acelerador**, sin la sobrecarga de lanzamiento y espera de XRT. No son mediciones de la duración interna del IP efectuadas directamente sobre la placa.

### 6.1 Latencia del IP: cosimulación C/RTL

**Latencia en ciclos de reloj:**

| Tamaño (N) | V0 — Secuencial | V1 — PIPELINE | V2 — ARRAY_PARTITION | V3 — DATAFLOW |
|---:|---:|---:|---:|---:|
| 10 | 478 | **442** | 590 | 548 |
| 100 | 1647 | **1251** | 1695 | 1887 |
| 500 | 6897 | **4905** | 6867 | 7779 |
| 1000 | 13461 | **9465** | 13311 | 15171 |

**Latencia equivalente en microsegundos (µs), utilizando el período objetivo HLS de 5 ns (200 MHz):**

| Tamaño (N) | V0 (µs) | V1 (µs) | V2 (µs) | V3 (µs) |
|---:|---:|---:|---:|---:|
| 10 | 2.390 | **2.210** | 2.950 | 2.740 |
| 100 | 8.235 | **6.255** | 8.475 | 9.435 |
| 500 | 34.485 | **24.525** | 34.335 | 38.895 |
| 1000 | 67.305 | **47.325** | 66.555 | 75.855 |

La conversión se realizó como `latencia (µs) = ciclos × 5 ns / 1000`. El período de **5 ns** corresponde al objetivo de los reportes HLS; **no debe confundirse con una verificación de la frecuencia efectiva del acelerador en la Kria**.

Las cuatro cosimulaciones Verilog terminaron con estado **`Pass`**. V1 obtuvo la menor latencia simulada para los cuatro tamaños. Para N=1000, V1 registró 9465 ciclos, frente a 13461 ciclos en V0 (una reducción aproximada del **29.7 %**). V3 obtuvo la mayor latencia simulada para N=100, 500 y 1000.

### 6.2 Síntesis HLS: latencia estimada, reloj y recursos

Los siguientes valores se extrajeron de la fila principal `softmax_hls` de los reportes `csynth.rpt` generados durante la construcción de los aceleradores (`tmp.hw`). Para V0–V2, la latencia estimada corresponde al tamaño máximo evaluado por el informe de síntesis; no debe sustituir la tabla de cosimulación por cada N.

| Métrica HLS | V0 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|
| Latencia estimada global (ciclos) | 12263 | 7265 | 7336 | No reportada (`DATAFLOW`) |
| Latencia estimada global (µs, a 5 ns) | 61.315 | 36.325 | 36.680 | No reportada |
| Intervalo global entre ejecuciones (ciclos) | 12264 | 7266 | 7337 | No reportado |
| Período objetivo de reloj (ns) | 5 | 5 | 5 | 5 |
| Frecuencia objetivo (MHz) | 200 | 200 | 200 | 200 |
| Slack informado (ns) | 0.00 | -0.29 | -0.29 | -0.29 |
| LUT | 4530 | 4860 | 6388 | 7008 |
| FF | 3011 | 3226 | 4957 | 5468 |
| BRAM | 6 | 6 | 8 | 10 |
| DSP | 9 | 9 | 9 | 9 |
| URAM | 0 | 0 | 0 | 0 |

**II (*initiation interval*) de los bucles:** es la cantidad de ciclos entre el inicio de iteraciones sucesivas. Aunque se solicite `PIPELINE II=1`, HLS puede obtener un valor mayor por dependencias u otras restricciones.

| Etapa / bucle | V0 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|
| Cálculo del máximo | 2 | 1 | 1 | 1 |
| Exponenciales | 9 (junto con suma) | 5 (junto con suma) | 5 (junto con suma) | 1 |
| Acumulación de la suma | 9 (junto con exponenciales) | 5 (junto con exponenciales) | 5 (junto con exponenciales) | 5 |
| Normalización | 1 | 1 | 1 | 1 |

V1 reduce la latencia estimada global un **40.8 %** respecto a V0. V2 conserva el II de V1 en los tres bucles, pero aumenta el uso de recursos. V3 introduce `DATAFLOW` y separa el cálculo de exponenciales y la acumulación; su informe **no proporciona una latencia global de síntesis**, aunque la cosimulación sí permite obtener las latencias por N. Los reportes de V1–V3 muestran un *slack* negativo de -0.29 ns, por lo que la restricción temporal debe verificarse con la implementación final.

En V2 y V3, `ARRAY_PARTITION cyclic factor=2` divide la entrada en dos interfaces AXI de lectura. Vitis HLS también informa que no pudo analizar el patrón de acceso para inferir ráfagas (*bursts*) de las entradas particionadas. En V3, el reporte registra además una **advertencia de estructura no canónica en la región `DATAFLOW`**. Estas observaciones deben considerarse al interpretar los resultados; por sí solas no permiten cuantificar el impacto de cada causa.

### 6.3 Procedencia y alcance de los reportes

**Archivos utilizados para la síntesis de cada versión**, relativos a la raíz del ejercicio:

```text
softmax_FP32_VN/HW/tmp.hw/softmax_hls/softmax_hls/softmax_hls/solution/syn/report/csynth.rpt
```

**Archivos con latencias de las transacciones C/RTL:**

```text
V0: softmax_FP32_V0/HW/softmax_hls/softmax_hls/hls/sim/verilog/softmax_hls.performance.result.transaction.xml
V1: softmax_FP32_V1/HW/softmax_hls/softmax_hls/hls/sim/verilog/softmax_hls.performance.result.transaction.xml
V2: softmax_FP32_V2/HW/softmax_hls/hls/sim/verilog/softmax_hls.performance.result.transaction.xml
V3: softmax_FP32_V3/HW/softmax_hls/hls/sim/verilog/softmax_hls.performance.result.transaction.xml
```


---

## 7. Alcance de las mediciones de potencia y energía

Las cifras de potencia y energía aparecen **en las cuatro tablas de la sección 5**, junto a los tiempos y al error numérico de la misma configuración. La lectura se realizó mediante el sensor **INA260** de la KV260, con un hilo que toma muestras durante la ejecución del lote.

La energía se estima mediante:

$$
E_{\mathrm{llamada}} = \frac{P_{\mathrm{media}}\, t_{\mathrm{lote}}}{5000}
$$

Aquí, si la potencia se expresa en **W** y el tiempo en **µs**, el resultado se obtiene en **µJ**. La energía por elemento es `E_llamada / N` (convertida a julios).

**Limitaciones:** los resultados corresponden al riel monitorizado por INA260; no se descontó potencia en reposo ni se aisló el consumo de la lógica FPGA. H2D y D2H se realizaron una vez por lote, y la energía por llamada se calculó mediante amortización.

---

## 8. Discusión de resultados FP32


- **Menor N:** para N=10, V0 registra 51.543 µs por llamada. Las diferencias entre V0 y V1 son muy pequeñas para este tamaño; V2 y V3 resultan ligeramente más lentas.
- **Mejora de V1:** en N=100, N=500 y N=1000, V1 reduce Kernel + XRT frente a V0 en aproximadamente **4.4 %, 12.6 % y 19.7 %**, respectivamente, para las corridas medidas.
- **V2:** es ligeramente mejor que V0 en N=500 y N=1000 (reducciones de ~1.4 % y ~4.6 % de Kernel + XRT), pero no supera V1. En N=1000 tiene mayor potencia media que V0; por ello, pese a su menor tiempo, la energía estimada por llamada es mayor (371.523 frente a 356.150 µJ).
- **V3:** registra tiempos más largos que V0 y V1, particularmente para N=500 y N=1000; en N=1000 tarda 127.572 µs por llamada, **26.9 % más que V0**. La presencia de `DATAFLOW` no garantiza aceleración: las dependencias entre etapas, el diseño de las transferencias y los costes de control condicionan el resultado.


---

## 9. Archivos de Softmax FP32

Cada combinación tiene un archivo `*_resumen.csv` y otro `*_llamadas.csv` con el mismo identificador de ejecución. Las rutas son relativas a la carpeta principal del ejercicio.

| Variante | N | Carpeta | Archivo oficial de resumen |
|---|---|---|---|
| V0 | 10 | `resultados_10/` | `resultados_softmax_fp32_V0_1795124016170_resumen.csv` |
| V0 | 100 | `resultados_100/` | `resultados_softmax_fp32_V0_1795124090940_resumen.csv` |
| V0 | 500 | `resultados_500/` | `resultados_softmax_fp32_V0_1795124114810_resumen.csv` |
| V0 | 1000 | `resultados_1000/` | `resultados_softmax_fp32_V0_1795124140088_resumen.csv` |
| V1 | 10 | `resultados_10/` | `resultados_softmax_fp32_V1_1795124230461_resumen.csv` |
| V1 | 100 | `resultados_100/` | `resultados_softmax_fp32_V1_1795124250323_resumen.csv` |
| V1 | 500 | `resultados_500/` | `resultados_softmax_fp32_V1_1795124267134_resumen.csv` |
| V1 | 1000 | `resultados_1000/` | `resultados_softmax_fp32_V1_1795124283765_resumen.csv` |
| V2 | 10 | `resultados_10/` | `resultados_softmax_fp32_V2_1795137646188_resumen.csv` |
| V2 | 100 | `resultados_100/` | `resultados_softmax_fp32_V2_1795136617370_resumen.csv` |
| V2 | 500 | `resultados_500/` | `resultados_softmax_fp32_V2_1795136644269_resumen.csv` |
| V2 | 1000 | `resultados_1000/` | `resultados_softmax_fp32_V2_1795136930481_resumen.csv` |
| V3 | 10 | `resultados_10/` | `resultados_softmax_fp32_V3_1795136959538_resumen.csv` |
| V3 | 100 | `resultados_100/` | `resultados_softmax_fp32_V3_1795136979205_resumen.csv` |
| V3 | 500 | `resultados_500/` | `resultados_softmax_fp32_V3_1795136993730_resumen.csv` |
| V3 | 1000 | `resultados_1000/` | `resultados_softmax_fp32_V3_1795137016687_resumen.csv` |

El archivo de llamadas correspondiente se obtiene sustituyendo `_resumen.csv` por `_llamadas.csv`. El identificador temporal permite distinguir repeticiones del experimento. Conservar los CSV anteriores identificados como fallidos únicamente como historial de diagnóstico, no combinarlos con las corridas validadas.

---

## 10. FP16 — campos reservados

**Pendiente de incorporar por la persona responsable de FP16.** No se atribuyen resultados, validaciones, optimizaciones ni métricas a FP16 en este documento.

### 10.1 Implementación y reproducción FP16

- Describir variantes V0–V3, pragmas utilizados y la interfaz de los `.xclbin` de FP16.
- Incluir los comandos de CSIM, CSYNTH, COSIM, compilación y empaquetado de FP16.
- Documentar la ruta del host FP16 y su formato de salida.

### 10.2 Resultados FP16 en la Kria — pendientes

Se reserva **una tabla por versión** con los mismos campos usados en FP32. Los datos de FP16 serán incorporados por la persona responsable; las rayas indican **pendiente**, no mediciones de valor cero.

#### V0 — Secuencial (FP16)

| N | H2D lote (µs) | Kernel + XRT (µs) | D2H lote (µs) | Tiempo total (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. | Estado |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 10 | — | — | — | — | — | — | — | — | — |
| 100 | — | — | — | — | — | — | — | — | — |
| 500 | — | — | — | — | — | — | — | — | — |
| 1000 | — | — | — | — | — | — | — | — | — |

#### V1 — PIPELINE (FP16)

| N | H2D lote (µs) | Kernel + XRT (µs) | D2H lote (µs) | Tiempo total (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. | Estado |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 10 | — | — | — | — | — | — | — | — | — |
| 100 | — | — | — | — | — | — | — | — | — |
| 500 | — | — | — | — | — | — | — | — | — |
| 1000 | — | — | — | — | — | — | — | — | — |

#### V2 — PIPELINE + ARRAY_PARTITION (FP16)

| N | H2D lote (µs) | Kernel + XRT (µs) | D2H lote (µs) | Tiempo total (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. | Estado |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 10 | — | — | — | — | — | — | — | — | — |
| 100 | — | — | — | — | — | — | — | — | — |
| 500 | — | — | — | — | — | — | — | — | — |
| 1000 | — | — | — | — | — | — | — | — | — |

#### V3 — DATAFLOW (FP16)

| N | H2D lote (µs) | Kernel + XRT (µs) | D2H lote (µs) | Tiempo total (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. | Estado |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 10 | — | — | — | — | — | — | — | — | — |
| 100 | — | — | — | — | — | — | — | — | — |
| 500 | — | — | — | — | — | — | — | — | — |
| 1000 | — | — | — | — | — | — | — | — | — |

### 10.4 Recursos y latencia HLS FP16

| Variante | Latencia IP HLS | II | Frecuencia | LUT | FF | DSP | BRAM |
|---|---|---|---|---|---|---|---|
| V0 | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente |
| V1 | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente |
| V2 | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente |
| V3 | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente |

---

## 11. Relación de archivos y reproducibilidad

| Ruta o archivo | Contenido |
|---|---|
| `softmax_FP32_V0/HW/` a `softmax_FP32_V3/HW/` | Código HLS, testbench, configuración y construcción de las cuatro versiones. |
| `softmax_FP32_VN/reportes/` | Reportes conservados de síntesis y cosimulación por versión, según disponibilidad. |
| `platform/` | Plataforma personalizada de Vitis para la KV260. |
| `FP32_VN/package.hw/` (Kria) | `.xclbin`, `.bit.bin`, `.dtbo` y `shell.json` utilizados en la ejecución. |
| `softmax_host.cpp` (Kria) | Programa XRT común, con soporte de entradas particionadas V2/V3, referencia FP64 y mediciones. |
| `softmax_host` (Kria) | Ejecutable compilado para arquitectura AArch64. |
| `resultados_{10,100,500,1000}/` | CSV de resumen y llamadas de cada versión. |
| `*_resumen.csv` | Latencias, H2D/D2H, total, potencia, energía, precisión y validación de una corrida. |
| `*_llamadas.csv` | Registro de 5000 duraciones individuales de Kernel + XRT. |

**Limitaciones y reproducibilidad:** conservar las rutas relativas, versiones de herramientas, configuración del hardware y correspondencia entre `.xclbin`, `.dtbo`, `.bit.bin` y `shell.json`; comprobar siempre que el acelerador activo coincide con el kernel que abre XRT.