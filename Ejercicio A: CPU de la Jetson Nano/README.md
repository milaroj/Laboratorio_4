# Ejercicio A: CPU de la Jetson Nano

Softmax numéricamente estable en FP16 y FP32 sobre la CPU ARM Cortex-A57 de la Jetson Nano, con exactitud medida contra una referencia en FP64, tiempo de la función y del programa completo, y potencia y energía con `tegrastats`.

---

## 1. Implementación

`softmax_cpu.cpp` calcula $m = \max_i x_i$ y $y_i = e^{x_i - m} / \sum_j e^{x_j - m}$ en tres pasadas sobre el vector:

1. Máximo.
2. `exp(x_i - m)`, guardado en `y`, y acumulación de la suma.
3. Normalización: `y_i / suma`.

La función es una plantilla con el tipo de almacenamiento `T` (`float` o FP16). Se marca `noinline` y cada llamada va seguida de una barrera del compilador, para que `-O3` no elimine llamadas repetidas.

**Decisión FP16.** Se usa `__fp16`. La Cortex-A57 es ARMv8.0-A; las instrucciones aritméticas FP16 llegaron con ARMv8.2-A. Por eso en esta CPU `__fp16` es solo formato de almacenamiento: GCC convierte cada valor a `float` (instrucción `fcvt`) antes de operar. Además libm no ofrece `exp` en FP16. Siguiendo el enunciado, la variante FP16 guarda entrada, salida y exponenciales intermedias en FP16, pero calcula el máximo, el `exp` y la suma en FP32.

**Entrada y referencia.** El vector se genera con `std::mt19937`, semilla 42, mapeado explícitamente a [−4, 4). No se usa `uniform_real_distribution` porque su algoritmo depende de la biblioteca estándar. Así cada plataforma recibe exactamente los mismos valores. La referencia es la misma softmax en `double` con `std::exp`, calculada desde la entrada FP32 original.

**Verificación** (en cada configuración):
- La suma de la salida es aproximadamente 1: tolerancia 1e-5 en FP32 y 5e-3 en FP16.
- Todos los valores están en [0, 1].
- La posición del máximo coincide con la referencia.
- Se reportan errores absoluto y relativo, máximo y medio, contra FP64.

---

## 2. Reproducción paso a paso

Comandos usados, en orden. Se trabaja en dos terminales: una en la laptop (raíz del repositorio) y otra con sesión SSH en la Jetson. La Jetson del curso se alcanza a través de un *jump host*; la cuenta `el5859` es compartida, así que cada grupo trabaja en su carpeta (`~/grupo-5`).

### 2.1 Copiar el código a la Jetson (laptop, raíz del repositorio)

Crear la carpeta de trabajo:

```bash
ssh -J el5859@51.161.8.220:9222 el5859@jetson-183 'mkdir -p ~/grupo-5/lab-cap4/ejA'
```

Copiar el código y los scripts. Se usa `tar` por `ssh` en lugar de `scp`, porque `scp` interpreta los dos puntos del nombre de la carpeta como un nombre de host:

```bash
tar -C "Ejercicio A: CPU de la Jetson Nano" -cf - softmax_cpu.cpp medir_cpu.sh energia.py | ssh -J el5859@51.161.8.220:9222 el5859@jetson-183 'tar -C ~/grupo-5/lab-cap4/ejA -xf -'
```

### 2.2 Compilar y probar (Jetson)

```bash
ssh -J el5859@51.161.8.220:9222 el5859@jetson-183
cd ~/grupo-5/lab-cap4/ejA
g++ -O3 -std=c++17 -march=native softmax_cpu.cpp -o softmax_cpu -lm
./softmax_cpu --precision fp16 --n 100 --reps 10000
./softmax_cpu --precision fp32 --n 100 --reps 10000
```

| Opción | Uso |
|---|---|
| `--measure funcion\|programa` | Qué se mide (por defecto `funcion`) |
| `--runs K` | Mediciones después de la corrida de calentamiento (por defecto 5) |
| `--seed S` | Semilla de la entrada (por defecto 42) |
| `--csv archivo.csv` | Agrega una fila de datos crudos por medición |

### 2.3 Estado del sistema e instrumento de potencia (Jetson)

Modo de potencia:

```bash
sudo nvpmodel -q
```

Revisión de `tegrastats`. Sin sudo no aparecen los rieles de potencia (`POM_5V_*`) ni las frecuencias de EMC y GPU, por eso todas las mediciones de energía lo usan con sudo:

```bash
timeout 3 tegrastats --interval 1000
sudo timeout 3 tegrastats --interval 1000
```

### 2.4 Mediciones completas (Jetson, ~4 min)

```bash
bash medir_cpu.sh
```

El script ejecuta, en orden:

1. **Estado del sistema** → `resultados/sistema_cpu.txt`.
2. **Reposo:** `tegrastats` durante 5 s sin carga.
3. **Tiempo de la función**, para `p` en {fp16, fp32} y `n` en {10, 100, 500, 1000}:
   ```bash
   ./softmax_cpu --precision $p --n $n --reps 10000 --csv resultados/cpu_funcion.csv
   ```
4. **Tiempo del programa completo:**
   ```bash
   ./softmax_cpu --precision $p --n $n --reps 10000 --measure programa --csv resultados/cpu_programa.csv
   ```
5. **Energía:** la función con las repeticiones necesarias para ~3 s por medición, calculadas con el tiempo del paso 3, mientras `tegrastats` graba:
   ```bash
   sudo tegrastats --interval 100 --logfile $PWD/resultados/tegrastats/tegra_cpu_${p}_funcion_n${n}.log --start
   ./softmax_cpu --precision $p --n $n --reps $REPS --csv resultados/cpu_energia.csv
   sudo tegrastats --stop
   ```
   Entre el arranque de `tegrastats` y el programa, y entre el programa y la parada, hay 1 s de margen. El registro necesita ruta absoluta: con `--start`, `tegrastats` corre en segundo plano desde `/`. La hora de inicio y de fin de cada registro se guarda en un archivo `.t0`.
6. **Potencia y energía por medición:**
   ```bash
   python3 energia.py --csv resultados/cpu_energia.csv --logs resultados/tegrastats \
       --reposo resultados/tegrastats/tegra_reposo.log \
       --salida resultados/cpu_energia_detalle.csv --resumen resultados/cpu_energia_resumen.csv
   ```

Toda la salida del programa queda además en `resultados/salida_cpu.txt`.

### 2.5 Traer los resultados al repositorio (laptop, raíz del repositorio)

```bash
ssh -J el5859@51.161.8.220:9222 el5859@jetson-183 'tar -C ~/grupo-5/lab-cap4/ejA -cf - softmax_cpu.cpp medir_cpu.sh energia.py resultados' | tar -C "Ejercicio A: CPU de la Jetson Nano" -xf -
```

---

## 3. Protocolo y entorno

| Parámetro | Valor |
|---|---|
| Modo de potencia | `MAXN` (modo 0, `sudo nvpmodel -q`) |
| Núcleos | 4 en línea (`0-3`); el programa ocupa 1 |
| Frecuencia CPU | 1479 MHz bajo carga en todas las muestras (máximo de MAXN); gobernador `schedutil` |
| Temperatura CPU | 28.5–33.5 °C durante las mediciones, sin limitación térmica |
| Compilador | g++ 7.5.0, `-O3 -std=c++17 -march=native` |
| Sistema | Ubuntu 18.04, L4T kernel 4.9.253-tegra |

Cada configuración tiene una corrida de calentamiento descartada y 5 mediciones; se reportan mediana, mínimo y máximo. El tiempo se mide con `std::chrono::steady_clock`.

- **Función:** solo las llamadas a `softmax`, 10 000 por medición. El valor reportado es el tiempo total entre 10 000.
- **Programa completo:** reservar memoria, generar la entrada, convertir a FP16/FP32, las 10 000 llamadas, calcular la referencia FP64 y verificar. No incluye el arranque del proceso ni la impresión.

---

## 4. Exactitud contra FP64

Todas las configuraciones pasan las tres verificaciones (suma ≈ 1, rango [0, 1] y posición del máximo). Los errores son idénticos en todas las mediciones, porque el cálculo es determinista.

| Precisión | N | Suma | \|suma − 1\| | Error abs. máx. | Error abs. medio | Error rel. máx. | Error rel. medio |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 10 | 1.0000696 | 6.96e-05 | 1.10e-04 | 3.26e-05 | 1.25e-03 | 4.31e-04 |
| FP16 | 100 | 0.9999301 | 6.99e-05 | 8.06e-05 | 5.69e-06 | 1.48e-03 | 4.54e-04 |
| FP16 | 500 | 1.0000235 | 2.35e-05 | 1.55e-05 | 9.58e-07 | 4.36e-03 | 6.32e-04 |
| FP16 | 1000 | 1.0000030 | 2.98e-06 | 7.79e-06 | 4.53e-07 | 1.00e-02 | 9.26e-04 |
| FP32 | 10 | 1.0000000 | 4.00e-09 | 9.21e-09 | 2.29e-09 | 2.81e-07 | 5.42e-08 |
| FP32 | 100 | 0.9999997 | 3.09e-07 | 2.93e-08 | 3.09e-09 | 6.00e-07 | 3.18e-07 |
| FP32 | 500 | 1.0000003 | 3.40e-07 | 5.17e-09 | 6.79e-10 | 6.39e-07 | 3.21e-07 |
| FP32 | 1000 | 1.0000004 | 4.46e-07 | 3.51e-09 | 4.46e-10 | 7.55e-07 | 4.35e-07 |

- El error relativo medio de FP16 (4–9e-4) está en el orden de su unidad de redondeo, 2⁻¹¹ ≈ 4.9e-4. En FP32 queda unas 1000 veces más abajo.
- El error relativo **máximo** de FP16 crece con N hasta 1 %. La causa es el rango de FP16, no la mantisa: al crecer N las salidas se achican (∝ 1/N) y muchas caen por debajo del menor normal de FP16, 6.1e-5. En esa zona subnormal el espaciado es fijo, 6e-8. Con N=1000 el 41 % de las salidas queda ahí (mínimo 2.6e-6), y con N=100 el 13 %.
- El error absoluto de FP16 baja con N por la misma razón: los valores son más pequeños.

---

## 5. Tiempo de ejecución

| Precisión | N | Función: µs/llamada (mediana) | mín – máx | ns/elemento | Programa completo: ms (mediana) | mín – máx | Programa − función |
|---|---:|---:|---|---:|---:|---|---:|
| FP16 | 10 | 0.3279 | 0.3265 – 0.3309 | 32.8 | 3.272 | 3.253 – 3.282 | −0.007 ms |
| FP16 | 100 | 3.0899 | 3.0873 – 3.0937 | 30.9 | 30.898 | 30.897 – 30.923 | 0.000 ms |
| FP16 | 500 | 15.3812 | 15.3784 – 15.3812 | 30.8 | 153.885 | 153.870 – 153.905 | +0.073 ms |
| FP16 | 1000 | 30.7335 | 30.7315 – 30.7381 | 30.7 | 307.445 | 307.435 – 308.871 | +0.110 ms |
| FP32 | 10 | 0.3077 | 0.3057 – 0.3088 | 30.8 | 3.091 | 3.071 – 3.135 | +0.014 ms |
| FP32 | 100 | 2.7457 | 2.7455 – 2.7465 | 27.5 | 27.475 | 27.472 – 27.510 | +0.017 ms |
| FP32 | 500 | 13.6680 | 13.6653 – 13.6704 | 27.3 | 136.788 | 136.761 – 137.081 | +0.108 ms |
| FP32 | 1000 | 27.3381 | 27.3368 – 27.3407 | 27.3 | 273.522 | 273.515 – 273.564 | +0.141 ms |

El programa completo usa 10 000 llamadas, igual que la función; "Programa − función" compara las medianas del tiempo total.

- **Escalamiento lineal:** a partir de N=100 el costo por elemento es constante, unos 27 ns en FP32, es decir ~40 ciclos a 1479 MHz. Lo domina la llamada a `expf`. Con N=10 pesan más el costo fijo de la llamada y de los tres bucles.
- **FP16 es más lento:** 6.6 % más lento con N=10 y 12.5 % con N ≥ 100. Sin aritmética FP16, cada lectura y escritura agrega una conversión, y el formato no ahorra cómputo. El ahorro de memoria tampoco ayuda: con N ≤ 1000 el vector cabe en la caché L1 de 32 KB.
- **Programa completo:** la diferencia con la función es menor que 0.5 % en todas las configuraciones. Con N ≤ 100 queda dentro de la variación entre mediciones; con N ≥ 500 es de unos 0.1 ms, sobre todo por la referencia FP64 y la verificación. Con 10 000 llamadas, el costo de preparar y verificar es despreciable frente al cómputo.

---

## 6. Potencia y energía

**Instrumento:** monitor INA3221 de la Jetson Nano leído con `sudo tegrastats --interval 100`. Cada lectura corresponde a un nivel distinto:

| Riel | Qué mide |
|---|---|
| `POM_5V_IN` | **Placa completa:** entrada de 5 V del módulo Jetson Nano (CPU, GPU, memoria y reguladores). No incluye la tarjeta portadora ni periféricos. |
| `POM_5V_CPU` | Riel de la CPU |
| `POM_5V_GPU` | Riel de la GPU: 0 mW en todas las muestras de este ejercicio |

**Método:**
- Se aumentan las repeticiones para que cada medición dure unos 3 s y el muestreo de 100 ms tenga resolución suficiente. El script calcula las repeticiones con el tiempo del paso anterior: de ~98 mil (N=1000) a ~9.8 millones (N=10).
- `energia.py` toma las muestras dentro de cada medición, descartando 0.2 s en cada borde, y promedia la potencia.
- Energía por llamada = potencia promedio × tiempo por llamada. La energía "sobre reposo" resta la potencia de la placa sin carga, medida 5 s antes de empezar.

**Reposo:** `POM_5V_IN` 966 mW, `POM_5V_CPU` 129 mW, `POM_5V_GPU` 0 mW.

| Precisión | N | t (µs) | P placa (mW) | P CPU (mW) | E placa (µJ) | E placa sobre reposo (µJ) | E riel CPU (µJ) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 10 | 0.3273 | 2318 | 911 | 0.759 | 0.443 | 0.298 |
| FP16 | 100 | 3.0872 | 2318 | 909 | 7.159 | 4.177 | 2.806 |
| FP16 | 500 | 15.384 | 2318 | 911 | 35.66 | 20.80 | 14.00 |
| FP16 | 1000 | 30.733 | 2320 | 911 | 71.29 | 41.62 | 27.99 |
| FP32 | 10 | 0.3063 | 2313 | 902 | 0.709 | 0.413 | 0.277 |
| FP32 | 100 | 2.7452 | 2318 | 909 | 6.364 | 3.713 | 2.495 |
| FP32 | 500 | 13.666 | 2318 | 902 | 31.68 | 18.48 | 12.33 |
| FP32 | 1000 | 27.353 | 2320 | 909 | 63.45 | 37.04 | 24.86 |

Medianas de 5 mediciones. Mínimos y máximos en `resultados/cpu_energia_resumen.csv`. Respecto de la mediana, las mediciones se desvían hasta 1.1 % en la potencia de la placa y hasta 2.5 % en el riel CPU (FP32 con N ≤ 100).

- La potencia es prácticamente la misma en todas las configuraciones: placa ~2.32 W (+1.35 W sobre reposo) y riel CPU ~0.91 W. Un núcleo ocupado a frecuencia fija consume lo mismo sin importar el formato o N, así que **la energía sigue al tiempo**.
- FP16 gasta 12 % más energía por llamada que FP32 con N ≥ 100 (71.3 frente a 63.5 µJ con N=1000). La energía dinámica por elemento es de ~42 nJ en FP16 y ~37 nJ en FP32.

---

## 7. Compromiso entre precisión, latencia, potencia y energía

En la CPU de la Jetson Nano, FP16 no ofrece ninguna ventaja:

- **Precisión:** el error medio es ~1000 veces mayor que en FP32, y el máximo llega a 1 % con N=1000 por los subnormales.
- **Latencia y energía:** es 12 % más lento y gasta 12 % más energía, porque la A57 tiene que convertir cada valor a FP32 para operar.
- **Potencia:** la misma que FP32.

FP16 solo sería útil aquí para reducir memoria o ancho de banda con vectores mucho más grandes que la caché. FP32 es la opción recomendada en esta CPU, y es la línea base contra la que se compara la GPU (Ejercicio B).

---

## 8. Archivos

| Archivo | Contenido |
|---|---|
| `softmax_cpu.cpp` | Implementación FP16/FP32, referencia FP64, verificación y medición |
| `medir_cpu.sh` | Corre todas las mediciones y graba `tegrastats` |
| `energia.py` | Cruza mediciones con muestras de potencia; calcula energía y resumen |
| `resultados/sistema_cpu.txt` | Modo de potencia, núcleos, gobernador, frecuencia máxima y compilador |
| `resultados/cpu_funcion.csv` | Datos crudos: tiempo de la función (10 000 reps) |
| `resultados/cpu_programa.csv` | Datos crudos: tiempo del programa completo (10 000 reps) |
| `resultados/cpu_energia.csv` | Datos crudos de las corridas de energía (~3 s por medición) |
| `resultados/cpu_energia_detalle.csv` | Potencia y energía por medición |
| `resultados/cpu_energia_resumen.csv` | Mediana, mínimo y máximo por configuración |
| `resultados/salida_cpu.txt` | Salida completa del programa en todas las corridas |
| `resultados/tegrastats/` | Registros crudos de `tegrastats` (`.log`) y su hora de inicio y fin (`.t0`) |
