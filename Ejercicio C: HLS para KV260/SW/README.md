# Softmax FP32 — Tablas de resultados V0–V3 | AMD Kria KV260

## Descripción

Tablas de resultados por implementación de Softmax FP32 para **V0–V3**, en el formato solicitado por el laboratorio. Por ahora se dispone de mediciones de **V0 (baseline)** sobre AMD Kria KV260, utilizando el host C++ y XRT 2.13.0. Las demás celdas se dejan pendientes.

## Condiciones de las mediciones principales

| Parámetro | Configuración |
|---|---|
| Plataforma | AMD Kria KV260 |
| Variante con datos medidos | V0 (baseline) |
| Precisión | FP32 |
| Tamaños de entrada | N = 10, 100, 500 y 1000 |
| Repeticiones medidas por N | 1000 |
| Calentamiento | 3 llamadas, excluidas del cronómetro del host |
| Host | C++ con `std::chrono::steady_clock` |
| Runtime | XRT 2.13.0 |
| Potencia | Sensor INA260 (`power1_input`) |
| Referencia numérica | Softmax calculado en FP64 |
| Estado de precisión | PASS en los cuatro tamaños |

## 1. Resultados — Formato solicitado en el laboratorio

**Precisión: FP32.** Se presenta una tabla por versión de Softmax: V0 (baseline), V1 (PIPELINE), V2 (PIPELINE + ARRAY_PARTITION) y V3 (PIPELINE + ARRAY_PARTITION + DATAFLOW). Las filas de CPU y GPU se mantienen para respetar el formato del enunciado; sus resultados están pendientes. Cuando se midan CPU y GPU, sus valores de referencia podrán compartirse entre las tablas, sin atribuirles cambios debidos a las variantes HLS.

En KV260, **Tiempo (µs)** corresponde al *tiempo total amortizado por llamada del host*, **no** a la latencia interna del IP. **Energía (µJ)** es energía estimada por llamada, y **J/elem.** es la energía estimada por elemento. Una raya (—) indica una medición aún no disponible.

### V0 — Baseline (FP32)

| Plataforma | N | Tiempo (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. |
|---|---:|---:|---:|---:|---:|---:|
| KV260 | 10 | 52.109 | 3.707 | 193.147 | 1.931 × 10⁻⁵ | 1.490 × 10⁻⁸ |
| KV260 | 100 | 57.206 | 3.687 | 210.900 | 2.109 × 10⁻⁶ | 1.051 × 10⁻⁸ |
| KV260 | 500 | 76.978 | 3.655 | 281.336 | 5.627 × 10⁻⁷ | 8.361 × 10⁻¹⁰ |
| KV260 | 1000 | 100.608 | 3.590 | 361.159 | 3.612 × 10⁻⁷ | 2.062 × 10⁻⁹ |

### V1 — PIPELINE (FP32)

| Plataforma | N | Tiempo (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. |
|---|---:|---:|---:|---:|---:|---:|
| KV260 | 10 | — | — | — | — | — |
| KV260 | 100 | — | — | — | — | — |
| KV260 | 500 | — | — | — | — | — |
| KV260 | 1000 | — | — | — | — | — |

### V2 — PIPELINE + ARRAY_PARTITION (FP32)

| Plataforma | N | Tiempo (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. |
|---|---:|---:|---:|---:|---:|---:|
| KV260 | 10 | — | — | — | — | — |
| KV260 | 100 | — | — | — | — | — |
| KV260 | 500 | — | — | — | — | — |
| KV260 | 1000 | — | — | — | — | — |

### V3 — PIPELINE + ARRAY_PARTITION + DATAFLOW (FP32)

| Plataforma | N | Tiempo (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. |
|---|---:|---:|---:|---:|---:|---:|
| KV260 | 10 | — | — | — | — | — |
| KV260 | 100 | — | — | — | — | — |
| KV260 | 500 | — | — | — | — | — |
| KV260 | 1000 | — | — | — | — | — |



## 2. Detalle adicional de V0 — Perfilado y mediciones separadas

### Descomposición temporal del host

**H2D y D2H se realizan una vez por lote**, mientras que `Kernel + XRT` se ejecuta 1000 veces. Por tanto, H2D y D2H en esta tabla son tiempos **por transferencia del lote**, no por llamada. La columna de tiempo total sí está amortizada por llamada.

| N | H2D (µs/lote) | Kernel + XRT (µs/llamada) | D2H (µs/lote) | Tiempo total (µs/llamada) |
|---:|---:|---:|---:|---:|
| 10 | 5.890 | 52.097 | 4.320 | 52.109 |
| 100 | 5.080 | 57.195 | 4.200 | 57.206 |
| 500 | 5.230 | 76.967 | 4.550 | 76.978 |
| 1000 | 5.370 | 100.597 | 4.370 | 100.608 |

### 2.1 Latencia interna del IP mediante ACCEL_MONITOR

La latencia del IP se calculó a partir de pares de eventos de inicio y fin en `device_trace_0.csv`, excluyendo **las primeras tres ejecuciones registradas**, bajo el supuesto de que representan las tres llamadas de calentamiento. No todas las llamadas aparecen en la traza: los promedios se calcularon únicamente con las ejecuciones efectivamente registradas.

| N | Latencia IP promedio (µs) | Ejecuciones registradas en la traza (incl. calentamiento) | Ejecuciones analizadas (sin calentamiento) | Llamadas esperadas (incl. calentamiento) |
|---:|---:|---:|---:|---:|
| 10 | 10.6427 | 953 | 950 | 1003 |
| 100 | 22.0686 | 967 | 964 | 1003 |
| 500 | 70.7074 | 956 | 953 | 1003 |
| 1000 | 132.4463 | 933 | 930 | 1003 |

### Resumen reportado directamente por XRT

Estos valores de `Average Time` **incluyen el calentamiento** y se muestran como referencia, distintos de los promedios calculados a partir de las trazas.

| N | `Number Of Calls` | `Total Time` (ms) | `Minimum Time` (µs) | `Average Time` (µs) | `Maximum Time` (µs) | Frecuencia reportada (MHz) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1003 | 10.6808 | 4.51005 | 10.6489 | 27.2203 | 99.999 |
| 100 | 1003 | 22.1726 | 19.7202 | 22.1062 | 42.1004 | 99.999 |
| 500 | 1003 | 70.9340 | 69.5107 | 70.7218 | 93.1609 | 99.999 |
| 1000 | 1003 | 132.8530 | 131.5610 | 132.4550 | 146.2610 | 99.999 |

### 2.2 Comparación host vs. latencia IP

| N | Kernel + XRT (µs/llamada) | Latencia IP (µs) | Observación |
|---:|---:|---:|---|
| 10 | 52.097 | 10.6427 | IP menor que host |
| 100 | 57.195 | 22.0686 | IP menor que host |
| 500 | 76.967 | 70.7074 | IP menor que host |
| 1000 | 100.597 | 132.4463 | **Discrepancia: IP mayor que host** |

**Advertencia metodológica.** Para N=1000 la latencia reportada mediante `ACCEL_MONITOR` es mayor que el tiempo `Kernel + XRT` medido desde el host. Esta discrepancia se reprodujo en otras corridas. Ambos valores se conservan **sin corregir ni reinterpretar artificialmente**; su causa no fue determinada. Los promedios del IP se basan en subconjuntos de trazas, no exactamente en las mismas 1000 llamadas que el host. No deben sumarse las columnas de latencia IP y `Kernel + XRT`, pues la primera corresponde conceptualmente a una parte de la segunda.

## 3. Repeticiones adicionales para N = 1000

Estas corridas fueron de comprobación y **no sustituyen** la corrida principal de 1000 repeticiones utilizada en las tablas anteriores.

| Corrida | Repeticiones medidas | Kernel + XRT (µs/llamada) | Tiempo total (µs/llamada) | Potencia (W) | Energía (µJ/llamada) | Latencia IP reportada por XRT, con calentamiento (µs) |
|---|---:|---:|---:|---:|---:|---:|
| Principal | 1000 | 100.597 | 100.608 | 3.590 | 361.159 | 132.455 |
| Verificación 2 | 1000 | 102.402 | 102.413 | 3.613 | 369.984 | 132.569 |
| Verificación 3 | 1500 | 100.249 | 100.257 | 3.579 | 358.775 | 132.306 |

En la tercera corrida XRT reportó 1503 llamadas y la traza contenía 1395 ejecuciones completas. La discrepancia entre host y XRT persistió.

## 4. Información de relojes

En los metadatos del archivo XCLBIN se observaron los siguientes valores:

| Elemento | Frecuencia |
|---|---:|
| Frecuencia solicitada con `v++ --clock.defaultFreqHz` | 250 MHz |
| `KERNEL_CLK` en `EMBEDDED_METADATA` | 199.998 MHz |
| `DATA_CLK` en `EMBEDDED_METADATA` | 99.999 MHz |
| Frecuencia mostrada en `summary.csv` de XRT | 99.999 MHz |

No se verificó la conexión física del reloj de `ACCEL_MONITOR` en el diseño implementado; por tanto, estos datos **no demuestran** que la discrepancia temporal se deba a una frecuencia de reloj incorrecta.

## 5. Archivos originales de resultados

Las corridas principales se guardaron con los siguientes nombres en la carpeta `SW`:

| N | CSV de resumen |
|---:|---|
| 10 | `resultados_softmax_fp32_V0_1794782882934_resumen.csv` |
| 100 | `resultados_softmax_fp32_V0_1794783234507_resumen.csv` |
| 500 | `resultados_softmax_fp32_V0_1794783367938_resumen.csv` |
| 1000 | `resultados_softmax_fp32_V0_1794784233708_resumen.csv` |

Para estas corridas también existen archivos `_llamadas.csv` con las mediciones individuales. Los archivos de perfilado de XRT (`summary.csv`, `device_trace_0.csv`, `xrt.run_summary`) se guardaron en las carpetas `perfilado_V0_N10`, `perfilado_V0_N100`, `perfilado_V0_N500` y `perfilado_V0_N1000`, respectivamente.

Corridas adicionales de N=1000:
- `resultados_softmax_fp32_V0_1794784668190_resumen.csv` (1000 repeticiones).
- `resultados_softmax_fp32_V0_1794784976294_resumen.csv` (1500 repeticiones).

---

**Estado:** precisión numérica validada (`PASS`) para todos los tamaños; latencia IP de N=1000 reportada con discrepancia metodológica pendiente de explicación.