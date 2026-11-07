# Ejercicio B: GPU CUDA en la Jetson Nano

Softmax numéricamente estable en FP16 (`__half`) y FP32 (`float`) sobre la GPU Maxwell integrada de la Jetson Nano. Se mide la exactitud contra FP64, y el tiempo se separa en cómputo del kernel, transferencias y lanzamiento, más la medición de extremo a extremo. La potencia y la energía se registran con `tegrastats`.

---

## 1. Implementación

### Bloques, hilos y memoria

| Aspecto | Decisión |
|---|---|
| Rejilla | 1 bloque. Con N ≤ 1000 el vector cabe en un bloque, y la GPU de la Nano tiene un solo SM (128 núcleos CUDA), así que más bloques no agregan paralelismo entre SM. |
| Hilos por bloque | N redondeado a múltiplo de 32, con máximo 256 (`--block`): 32 hilos para N=10, 128 para N=100 y 256 para N=500 y N=1000 (4 elementos por hilo). |
| Reparto | El hilo `t` procesa `i = t, t+256, t+512, ...`, así los hilos consecutivos leen direcciones consecutivas (acceso coalescido). |
| Memoria global | `x` (entrada) e `y` (salida y valores intermedios de `exp`). |
| Memoria compartida | 32 `float` (128 B) para los resultados parciales de cada warp. |
| Memoria del host | Fija (`cudaMallocHost`), para que las copias no pasen por un búfer intermedio. |

### Etapas, en un solo kernel

1. **Máximo:** cada hilo calcula el máximo de sus elementos. Luego se reduce dentro de cada warp con `__shfl_down_sync`, los 8 resultados de warp pasan por memoria compartida, y el warp 0 hace la reducción final y la reparte a todos los hilos.
2. **Exponencial y suma:** cada hilo calcula `e_i = exp(x_i − m)`, lo guarda en `y` y acumula su suma parcial, que se reduce igual que el máximo.
3. **Normalización:** `y_i = e_i / suma`. Cada hilo relee los valores que él mismo escribió, así que no hace falta otra sincronización.

Unir las tres etapas en un kernel evita dos lanzamientos extra y dos viajes por memoria global entre kernels. Como se ve en la sección 6, el lanzamiento es el costo dominante.

### Decisión FP16

A diferencia de la CPU (Ejercicio A), la GPU Maxwell (sm_53) sí tiene aritmética `__half` nativa. Por eso la resta, el exponencial (`hexp`) y la división (`__hdiv`) se hacen en `__half`. El ensamblador generado (`cuobjdump -sass`) contiene instrucciones `HADD2` y `HFMA2`; `hexp` usa internamente la unidad especial `MUFU.EX2` en FP32, con conversiones de ida y vuelta.

Hay dos excepciones:
- El máximo se compara convirtiendo a `float`. Es exacto: el resultado es el mismo valor `__half`.
- La suma se acumula en `float` y se redondea a `__half` solo para la división. Una suma en `__half` de 1000 términos perdería mucha precisión: el espaciado de `__half` cerca de 1 es 9.8e-4.

### Qué mide cada modo

| `--measure` | Región medida | Herramienta |
|---|---|---|
| `kernel` | 10 000 lanzamientos consecutivos del kernel, con los datos ya en la GPU | Eventos CUDA (`cudaEventRecord`/`cudaEventElapsedTime`) |
| `end-to-end` | 10 000 repeticiones de copia H→D + kernel + copia D→H. La copia D→H es síncrona y espera al kernel, así que incluye la sincronización. | `std::chrono::steady_clock` |
| nvprof | Duración en la GPU de cada kernel y cada copia, y tiempo de las llamadas a la API | `sudo nvprof -u us --csv` |

La entrada y la referencia son las del Ejercicio A: `mt19937` con semilla 42 en [−4, 4) y softmax en `double`. La verificación también es la misma: suma ≈ 1, rango [0, 1], posición del máximo, y errores absoluto y relativo.

---

## 2. Reproducción paso a paso

Comandos usados, en orden. La terminal de la laptop está en la raíz del repositorio.

### 2.1 Copiar el código a la Jetson (laptop)

```bash
tar -C "Ejercicio B: GPU CUDA en la Jetson Nano" -cf - softmax_cuda.cu medir_gpu.sh energia.py nvprof_resumen.py | ssh -J el5859@51.161.8.220:9222 el5859@jetson-183 'mkdir -p ~/grupo-5/lab-cap4/ejB && tar -C ~/grupo-5/lab-cap4/ejB -xf -'
```

### 2.2 Compilar y probar (Jetson)

`ServerAliveInterval` evita que la sesión se cierre por inactividad durante las mediciones largas.

```bash
ssh -o ServerAliveInterval=30 -J el5859@51.161.8.220:9222 el5859@jetson-183
cd ~/grupo-5/lab-cap4/ejB
nvcc -O3 -std=c++14 -arch=sm_53 softmax_cuda.cu -o softmax_cuda
```

Se agrega `-arch=sm_53` al comando del enunciado. Sin esa opción, nvcc 10.2 compila para sm_30, que no tiene las operaciones aritméticas `__half`.

```bash
./softmax_cuda --precision fp16 --n 100 --reps 10000 --measure kernel
./softmax_cuda --precision fp16 --n 100 --reps 10000 --measure end-to-end
```

| Opción | Uso |
|---|---|
| `--measure kernel\|end-to-end` | Qué se mide (por defecto `kernel`) |
| `--runs K` | Mediciones después del calentamiento (por defecto 5) |
| `--block B` | Hilos máximos por bloque, múltiplo de 32 (por defecto 256) |
| `--seed S` | Semilla de la entrada (por defecto 42) |
| `--csv archivo.csv` | Agrega una fila de datos crudos por medición |

### 2.3 Fijar los relojes (Jetson)

Con el gobernador por defecto, la GPU se quedó en 384–461 MHz, porque los kernels son tan cortos que la utilización parece baja. Para medir siempre a la misma frecuencia se fijaron CPU, GPU y memoria al máximo de MAXN, y se restauraron al terminar, porque la Jetson es compartida:

```bash
sudo nvpmodel -q
sudo jetson_clocks --store ~/grupo-5/lab-cap4/ejB/relojes_originales.conf
sudo jetson_clocks && sudo jetson_clocks --show | tee relojes_fijos.txt
```

### 2.4 Mediciones completas (Jetson, ~10 min)

```bash
bash medir_gpu.sh
```

El script ejecuta, en orden, para `m` en {kernel, end-to-end}, `p` en {fp16, fp32} y `n` en {10, 100, 500, 1000}:

1. **Estado del sistema** → `resultados/sistema_gpu.txt`.
2. **Reposo:** `tegrastats` durante 5 s sin carga.
3. **Tiempos instrumentados:**
   ```bash
   ./softmax_cuda --precision $p --n $n --reps 10000 --measure $m --csv resultados/gpu_$m.csv
   ```
4. **Perfilador:**
   ```bash
   sudo /usr/local/cuda/bin/nvprof -u us --csv --log-file resultados/nvprof/nvprof_${p}_${m}_n${n}.csv \
       ./softmax_cuda --precision $p --n $n --reps 10000 --measure $m
   ```
5. **Energía:** las repeticiones necesarias para ~3 s por medición, calculadas con el tiempo del paso 3, mientras `tegrastats` graba. El procedimiento es el mismo del Ejercicio A, con 1 s de margen y la hora de inicio y de fin en `.t0`:
   ```bash
   sudo tegrastats --interval 100 --logfile $PWD/resultados/tegrastats/tegra_gpu_${p}_${m}_n${n}.log --start
   ./softmax_cuda --precision $p --n $n --reps $REPS --measure $m --csv resultados/gpu_energia.csv
   sudo tegrastats --stop
   ```
6. **Tablas:**
   ```bash
   python3 nvprof_resumen.py --logs resultados/nvprof --salida resultados/gpu_nvprof_resumen.csv
   python3 energia.py --csv resultados/gpu_energia.csv --logs resultados/tegrastats \
       --reposo resultados/tegrastats/tegra_reposo.log \
       --salida resultados/gpu_energia_detalle.csv --resumen resultados/gpu_energia_resumen.csv
   ```

### 2.5 Restaurar los relojes (Jetson)

```bash
sudo jetson_clocks --restore ~/grupo-5/lab-cap4/ejB/relojes_originales.conf && sudo jetson_clocks --show
```

### 2.6 Traer los resultados al repositorio (laptop)

```bash
ssh -J el5859@51.161.8.220:9222 el5859@jetson-183 'tar -C ~/grupo-5/lab-cap4/ejB -cf - softmax_cuda.cu medir_gpu.sh energia.py nvprof_resumen.py relojes_fijos.txt relojes_originales.conf resultados' | tar -C "Ejercicio B: GPU CUDA en la Jetson Nano" -xf -
```

---

## 3. Protocolo y entorno

| Parámetro | Valor |
|---|---|
| GPU | NVIDIA Tegra X1, Maxwell sm_53, 1 SM (128 núcleos CUDA) |
| Software | CUDA 10.2 (runtime y driver), JetPack sobre Ubuntu 18.04 |
| Modo de potencia | `MAXN` (modo 0) |
| Relojes durante las mediciones | `jetson_clocks`: GPU 921.6 MHz, CPU 1479 MHz en los 4 núcleos, EMC 1600 MHz; estados de reposo de la CPU desactivados (`relojes_fijos.txt`) |
| Frecuencia de GPU registrada | 921.6 MHz en todas las mediciones (CSV y `tegrastats`) |
| Temperatura | GPU 32.5–46 °C, CPU hasta 48 °C; sin limitación térmica |

Cada configuración tiene una corrida de calentamiento descartada y 5 mediciones de 10 000 repeticiones. Se reportan mediana, mínimo y máximo.

**Observaciones:**
- Una corrida manual previa al script (FP16, N=100, extremo a extremo) dio 97.9 µs por llamada, contra 77–79 µs en las corridas del script. No se identificó la causa. La Jetson es compartida, así que no se descarta la carga de otro usuario. Se reportan los datos del script, que coinciden entre las corridas de tiempo y las de energía.
- FP16 `kernel` con N=1000 tuvo tres mediciones lentas (32.6, 43.4 y 26.7 µs) contra ~23.6 µs en las otras dos. La corrida de energía de esa misma configuración, de ~3 s por medición, dio 23.0 µs, igual que el resto.

---

## 4. Exactitud contra FP64

Las 16 combinaciones de precisión, N y modo pasan las tres verificaciones. Los errores no dependen del modo de medición.

| Precisión | N | Hilos | Suma | \|suma − 1\| | Error abs. máx. | Error abs. medio | Error rel. máx. | Error rel. medio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 10 | 32 | 1.0002151 | 2.15e-04 | 2.15e-04 | 4.70e-05 | 2.91e-03 | 8.59e-04 |
| FP16 | 100 | 128 | 0.9997867 | 2.13e-04 | 8.06e-05 | 6.24e-06 | 3.27e-03 | 9.56e-04 |
| FP16 | 500 | 256 | 0.9998789 | 1.21e-04 | 1.55e-05 | 1.14e-06 | 7.09e-03 | 9.66e-04 |
| FP16 | 1000 | 256 | 0.9996440 | 3.56e-04 | 9.24e-06 | 6.06e-07 | 1.29e-02 | 1.27e-03 |
| FP32 | 10 | 32 | 1.0000000 | 2.40e-08 | 1.17e-08 | 3.61e-09 | 2.25e-07 | 8.93e-08 |
| FP32 | 100 | 128 | 1.0000000 | 1.00e-09 | 5.65e-09 | 3.60e-10 | 3.34e-07 | 1.08e-07 |
| FP32 | 500 | 256 | 1.0000000 | 2.70e-08 | 1.42e-09 | 9.42e-11 | 3.54e-07 | 1.01e-07 |
| FP32 | 1000 | 256 | 1.0000000 | 3.50e-08 | 7.42e-10 | 4.99e-11 | 3.81e-07 | 1.01e-07 |

**Comparación con la CPU (Ejercicio A):**
- **FP16 en GPU tiene más error que en CPU:** error relativo medio de 0.9–1.3e-3, contra 0.4–0.9e-3. En la GPU la resta, el `exp` y la división redondean a `__half` en cada paso; en la CPU solo se redondea al guardar. Además, la suma se divide ya redondeada a `__half`, lo que sesga todas las salidas por igual. Por eso \|suma − 1\| es de 1–4e-4, contra 3e-6–7e-5 en la CPU.
- **FP32 en GPU es más exacto que en CPU con N grande:** error relativo medio de 1.0e-7 contra 4.4e-7 con N=1000. La suma de la GPU es una reducción en árbol (por pares), cuyo error crece con log N, mientras que en la CPU la suma es secuencial y su error crece con N.
- El error relativo máximo de FP16 vuelve a crecer con N (1.3 % con N=1000), por la misma causa que en la CPU: muchas salidas caen en el rango subnormal de FP16.

---

## 5. Tiempo instrumentado

| Precisión | N | `kernel`: µs/lanzamiento (mediana) | mín – máx | `end-to-end`: µs/llamada (mediana) | mín – máx | CPU (Ej. A) µs | GPU e2e / CPU |
|---|---:|---:|---|---:|---|---:|---:|
| FP16 | 10 | 23.31 | 22.96 – 23.37 | 77.40 | 77.12 – 77.58 | 0.328 | 236× |
| FP16 | 100 | 23.40 | 23.33 – 23.54 | 77.98 | 76.98 – 79.08 | 3.090 | 25× |
| FP16 | 500 | 23.29 | 23.19 – 23.33 | 78.50 | 78.42 – 78.61 | 15.38 | 5.1× |
| FP16 | 1000 | 26.66 | 23.59 – 43.41 | 79.37 | 78.93 – 81.35 | 30.73 | 2.6× |
| FP32 | 10 | 22.93 | 22.81 – 22.95 | 76.47 | 75.79 – 76.60 | 0.308 | 249× |
| FP32 | 100 | 23.22 | 23.11 – 23.32 | 78.93 | 78.77 – 80.65 | 2.746 | 29× |
| FP32 | 500 | 23.41 | 23.26 – 23.47 | 78.67 | 78.27 – 79.08 | 13.67 | 5.8× |
| FP32 | 1000 | 23.39 | 23.23 – 23.47 | 84.13 | 83.85 – 84.58 | 27.34 | 3.1× |

- Los dos modos son **prácticamente constantes con N**: ~23 µs por lanzamiento y ~77–84 µs de extremo a extremo. El tiempo no depende del cálculo sino de costos fijos.
- En todos los tamaños medidos, la GPU de extremo a extremo es más lenta que la CPU: 2.6× con N=1000 y más de 200× con N=10.

---

## 6. Cómputo, transferencias y lanzamiento (nvprof)

| Precisión | N | Kernel en GPU: prom. (µs) | mín – máx | Copia H→D (µs) | Copia D→H (µs) | `cudaLaunchKernel` con nvprof (µs) | `cudaMemcpy` con nvprof (µs) |
|---|---:|---:|---|---:|---:|---:|---:|
| FP16 | 10 | 3.629 | 3.593 – 6.823 | 0.714 | 0.468 | 40.8 | 39.3 |
| FP16 | 100 | 3.516 | 3.489 – 6.250 | 0.761 | 0.499 | 40.8 | 41.7 |
| FP16 | 500 | 4.350 | 4.270 – 6.979 | 0.837 | 0.562 | 42.4 | 43.8 |
| FP16 | 1000 | 5.844 | 5.781 – 8.385 | 0.919 | 0.648 | 44.2 | 46.7 |
| FP32 | 10 | 3.512 | 3.437 – 6.667 | 0.717 | 0.471 | 40.8 | 39.9 |
| FP32 | 100 | 3.395 | 3.333 – 6.197 | 0.790 | 0.526 | 40.8 | 41.4 |
| FP32 | 500 | 4.288 | 4.218 – 6.667 | 0.917 | 0.648 | 40.6 | 43.3 |
| FP32 | 1000 | 5.845 | 5.729 – 8.021 | 1.086 | 0.805 | 41.6 | 54.2 |

Datos de las corridas `end-to-end` con nvprof: 60 000 kernels y 60 001 copias en cada sentido por configuración. El modo `kernel` da las mismas duraciones de kernel (`resultados/gpu_nvprof_resumen.csv`). Las columnas de la API miden el lado de la CPU e incluyen el costo agregado por el perfilador: sin nvprof, el lanzamiento consecutivo cuesta ~23 µs (sección 5), y con nvprof `cudaLaunchKernel` sube a 37–44 µs.

**Desglose de una llamada de extremo a extremo (FP32):**

| N | Kernel (GPU) | Copias (GPU) | Trabajo en GPU | Extremo a extremo | Resto: API, driver y sincronización | % del tiempo con la GPU trabajando |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 3.51 µs | 1.19 µs | 4.70 µs | 76.47 µs | 71.8 µs | 6 % |
| 100 | 3.40 µs | 1.32 µs | 4.71 µs | 78.93 µs | 74.2 µs | 6 % |
| 500 | 4.29 µs | 1.57 µs | 5.85 µs | 78.67 µs | 72.8 µs | 7 % |
| 1000 | 5.85 µs | 1.89 µs | 7.74 µs | 84.13 µs | 76.4 µs | 9 % |

- **Cómputo:** el kernel tarda 3.4–5.8 µs. Con N=1000 la GPU calcula la softmax **4.7 veces más rápido que la CPU** (5.8 frente a 27.3 µs). Con N ≤ 100 el kernel dura lo mismo para cualquier N: domina la latencia fija de las reducciones y las barreras.
- **Transferencias:** con ≤ 4 KB por vector, cada copia dura apenas 0.5–1.1 µs en la GPU.
- **Lanzamiento y sincronización:** son más del 90 % del tiempo. En modo `kernel` la GPU calcula ~3.5–5.8 µs de cada ~23 µs, y el resto espera el siguiente lanzamiento; por eso `tegrastats` marca una utilización (GR3D) de solo 20–30 %. De extremo a extremo hay además dos `cudaMemcpy` síncronos por llamada, cada uno con su paso por el driver. Estimado restando el modo `kernel`, los dos `cudaMemcpy` cuestan ~55 µs sin perfilador.
- **FP16 contra FP32:** el kernel dura lo mismo (+3 % en FP16 con N ≤ 100, igual con N=1000). Las operaciones `__half` son escalares, sin vectorizar con `half2`, y `hexp` y `__hdiv` pasan por FP32 internamente. Las copias FP16 son 15–20 % más rápidas con N=1000 porque mueven la mitad de bytes, pero eso es menos de 0.4 µs de una llamada de ~80 µs.

---

## 7. Potencia y energía

Mismo instrumento y método del Ejercicio A:

| Riel | Qué mide |
|---|---|
| `POM_5V_IN` | **Placa completa:** módulo Jetson Nano, sin la tarjeta portadora |
| `POM_5V_GPU` | **Acelerador:** riel de la GPU |
| `POM_5V_CPU` | Riel de la CPU, que lanza los kernels y espera |

Repeticiones para ~3 s por medición, muestras de `tegrastats` cada 100 ms y 0.2 s descartados en cada borde.

**Reposo con `jetson_clocks`:** `POM_5V_IN` 1872 mW, `POM_5V_CPU` 292 mW, `POM_5V_GPU` 125 mW. Es mayor que el reposo del Ejercicio A (966 mW), porque los relojes fijos y la CPU sin estados de reposo consumen ~0.9 W más aun sin carga.

| Precisión | Modo | N | t (µs) | P placa (mW) | P CPU (mW) | P GPU (mW) | GR3D | E placa (µJ) | E placa sobre reposo (µJ) | E GPU (µJ) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | kernel | 10 | 22.95 | 4165 | 2047 | 570 | 20 % | 95.6 | 52.6 | 13.1 |
| FP16 | kernel | 100 | 23.31 | 4198 | 2040 | 611 | 20 % | 97.8 | 54.2 | 14.2 |
| FP16 | kernel | 500 | 22.75 | 4292 | 2075 | 693 | 25 % | 97.8 | 55.2 | 15.8 |
| FP16 | kernel | 1000 | 23.05 | 4353 | 2066 | 732 | 30 % | 100.2 | 57.1 | 16.9 |
| FP32 | kernel | 10 | 23.00 | 4231 | 2077 | 611 | 20 % | 97.5 | 54.4 | 14.1 |
| FP32 | kernel | 100 | 22.99 | 4273 | 2080 | 652 | 20 % | 98.3 | 55.3 | 15.0 |
| FP32 | kernel | 500 | 22.88 | 4355 | 2084 | 722 | 25 % | 99.7 | 56.9 | 16.7 |
| FP32 | kernel | 1000 | 22.99 | 4426 | 2087 | 773 | 30 % | 101.9 | 58.8 | 17.8 |
| FP16 | end-to-end | 10 | 78.69 | 4044 | 1943 | 560 | 7 % | 317.9 | 170.6 | 44.1 |
| FP16 | end-to-end | 100 | 77.51 | 4076 | 1961 | 571 | 7 % | 316.3 | 171.5 | 44.3 |
| FP16 | end-to-end | 500 | 78.45 | 4097 | 1952 | 571 | 9 % | 321.5 | 174.3 | 44.8 |
| FP16 | end-to-end | 1000 | 78.97 | 4134 | 1961 | 612 | 11 % | 326.4 | 178.6 | 48.3 |
| FP32 | end-to-end | 10 | 77.63 | 4105 | 1978 | 571 | 7 % | 317.7 | 172.5 | 44.3 |
| FP32 | end-to-end | 100 | 78.33 | 4133 | 1992 | 571 | 7 % | 324.3 | 177.5 | 44.7 |
| FP32 | end-to-end | 500 | 79.61 | 4130 | 1956 | 612 | 9 % | 328.6 | 179.6 | 48.7 |
| FP32 | end-to-end | 1000 | 80.00 | 4176 | 1963 | 645 | 11 % | 333.7 | 184.1 | 51.8 |

Medianas de 5 mediciones; mínimos y máximos en `resultados/gpu_energia_resumen.csv`. En la mayoría de configuraciones las mediciones se desvían menos de 3 % de la mediana. Cinco configuraciones tienen una medición atípica, hasta 10 % más lenta y 6 % más energía, que no afecta la mediana.

- **La CPU consume más que la GPU:** ~2 W en el riel de CPU contra 0.56–0.77 W en el de GPU. La CPU no descansa: lanza kernels y espera las copias. La potencia de la GPU sube con N y con la utilización.
- **Energía por llamada casi constante con N**, igual que el tiempo: ~96–102 µJ por lanzamiento y ~316–334 µJ de extremo a extremo (placa).
- **FP16 contra FP32:** hasta 2.5 % menos energía en FP16 (y ninguna diferencia con N=10 de extremo a extremo), sobre todo por la menor potencia de la GPU. Es una diferencia del orden de la variación entre mediciones.

**Comparación con la CPU (Ejercicio A).** La energía sobre el reposo propio es la comparación más justa, porque los reposos difieren por `jetson_clocks`:

| N | CPU FP32 (µJ) | GPU FP32 extremo a extremo (µJ) | GPU / CPU | GPU FP32 por lanzamiento (µJ) | GPU / CPU |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.41 | 172.5 | 418× | 54.4 | 132× |
| 100 | 3.71 | 177.5 | 48× | 55.3 | 15× |
| 500 | 18.48 | 179.6 | 9.7× | 56.9 | 3.1× |
| 1000 | 37.04 | 184.1 | 5.0× | 58.8 | 1.6× |

---

## 8. Compromiso entre precisión, latencia, potencia y energía

- **GPU contra CPU:** con N ≤ 1000 la CPU gana en latencia y en energía en todos los tamaños medidos, aunque el kernel calcule 4.7 veces más rápido con N=1000. La GPU gasta más del 90 % del tiempo en lanzamiento, driver y sincronización, y mientras tanto la placa consume ~4.1–4.4 W, contra ~2.3 W en la CPU.
  - Si la GPU mantuviera ~77 µs fijos más lo que crecen el kernel y las copias (~3.4 ns por elemento según nvprof), el extremo a extremo igualaría a la CPU (27 ns por elemento) alrededor de **N ≈ 3000**. Es una extrapolación fuera del rango medido.
  - Aun sin transferencias, el lanzamiento consecutivo (~23 µs) iguala a la CPU cerca de N ≈ 850.
- **Para que la GPU convenga** hay que amortizar los costos fijos: vectores mucho más grandes, procesar muchos vectores en un solo lanzamiento (un bloque por vector), o encadenar trabajo sin copias intermedias. En la Jetson, la CPU y la GPU comparten la memoria física, así que la memoria *zero-copy* (`cudaHostAlloc` mapeada) o la memoria unificada eliminarían las copias explícitas.
- **FP16 contra FP32 en GPU:** el tiempo y la energía son casi iguales con estos tamaños, porque el cómputo no es el cuello de botella. FP16 sí empeora la exactitud: el error relativo medio es ~9 000–13 000 veces mayor que en FP32, y el máximo llega a 1.3 % con N=1000. Aquí FP16 solo tendría sentido con datos mucho más grandes, limitados por el ancho de banda, y vectorizando con `half2`.

---

## 9. Archivos

| Archivo | Contenido |
|---|---|
| `softmax_cuda.cu` | Kernel de softmax FP16/FP32 con reducciones por bloque, referencia FP64, verificación y medición |
| `medir_gpu.sh` | Corre todas las mediciones: tiempos, nvprof y energía con `tegrastats` |
| `energia.py` | Cruza las mediciones con las muestras de potencia (copia del Ej. A con columnas de GPU) |
| `nvprof_resumen.py` | Resume los registros de nvprof: kernel, copias y API |
| `relojes_fijos.txt` | Estado de los relojes con `jetson_clocks` durante las mediciones |
| `relojes_originales.conf` | Configuración de relojes guardada antes de fijarlos (usada para restaurar) |
| `resultados/sistema_gpu.txt` | Modo de potencia, frecuencia de GPU, gobernador y versión de CUDA |
| `resultados/gpu_kernel.csv` | Datos crudos: modo `kernel` (10 000 reps) |
| `resultados/gpu_end-to-end.csv` | Datos crudos: modo `end-to-end` (10 000 reps) |
| `resultados/nvprof/` | Registros crudos de nvprof por configuración |
| `resultados/gpu_nvprof_resumen.csv` | Resumen de nvprof |
| `resultados/gpu_energia.csv` | Datos crudos de las corridas de energía (~3 s por medición) |
| `resultados/gpu_energia_detalle.csv` | Potencia y energía por medición |
| `resultados/gpu_energia_resumen.csv` | Mediana, mínimo y máximo por configuración |
| `resultados/salida_gpu.txt` | Salida completa del programa en todas las corridas |
| `resultados/tegrastats/` | Registros crudos de `tegrastats` y su hora de inicio y fin |
