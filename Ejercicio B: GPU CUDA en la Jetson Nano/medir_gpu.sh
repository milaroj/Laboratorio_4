#!/bin/bash
# medir_gpu.sh — Laboratorio 4, Ejercicio B: corre todas las mediciones de la GPU.
#
#  1. Reposo: registra tegrastats 5 s sin carga (potencia de referencia).
#  2. Tiempos instrumentados: N en {10,100,500,1000}, FP16 y FP32, 10 000 reps,
#     medida "kernel" (eventos CUDA) y "end-to-end" (reloj del host).
#  3. Perfilador: nvprof sobre las mismas corridas (kernel, copias y llamadas a la API).
#  4. Energía: las mismas configuraciones con las repeticiones necesarias para que
#     cada medición dure ~3 s, mientras tegrastats muestrea cada 100 ms.
#  5. energia.py y nvprof_resumen.py generan las tablas.
#
# Uso, en la Jetson y desde la carpeta del ejercicio:  bash medir_gpu.sh
# Pide la contraseña de sudo una vez (tegrastats y nvprof la necesitan).
set -euo pipefail

BIN=./softmax_cuda
RES="$PWD/resultados"            # absoluta: tegrastats --start trabaja desde /
TEG=$RES/tegrastats
NVP=$RES/nvprof
NVPROF=/usr/local/cuda/bin/nvprof
INTERVALO=100                    # ms entre muestras de tegrastats
NS=(10 100 500 1000)
PRECS=(fp16 fp32)
MEDIDAS=(kernel end-to-end)
REPS_TIEMPO=10000
DURACION=3                       # segundos por medición en las corridas de energía

mkdir -p "$TEG" "$NVP"
SALIDA=$RES/salida_gpu.txt
: > "$SALIDA"
sudo -v

# Repeticiones para que una medición dure ~DURACION s, según el tiempo por
# llamada medido en el paso 2 (columna tiempo_llamada_us de gpu_<medida>.csv).
reps_energia() {
    awk -F, -v p="$2" -v n="$3" -v d="$DURACION" '
        NR == 1 { for (i = 1; i <= NF; i++) if ($i == "tiempo_llamada_us") col = i; next }
        $2 == p && $4 == n { s += $col; c++ }
        END { printf "%d\n", d * 1e6 / (s / c) }' "$RES/gpu_$1.csv"
}

# Graba tegrastats mientras corre un comando (igual que en el Ejercicio A).
con_tegrastats() {
    local log=$1; shift
    sudo tegrastats --stop > /dev/null 2>&1 || true
    sudo rm -f "$log"
    local t0; t0=$(date +%s.%N)
    sudo tegrastats --interval "$INTERVALO" --logfile "$log" --start
    sleep 1
    "$@" | tee -a "$SALIDA"
    sleep 1
    sudo tegrastats --stop
    local t1; t1=$(date +%s.%N)
    echo "$t0 $t1 $INTERVALO" > "$log.t0"
}

leer() { cat "$1" 2> /dev/null || echo "?"; }
GPUFREQ=/sys/class/devfreq/57000000.gpu
echo "== Estado del sistema"
{
    echo "fecha: $(date -Is)"
    echo "host: $(hostname)"
    sudo nvpmodel -q 2> /dev/null | grep -v WARN || true
    echo "frecuencia GPU actual / min / max (Hz): $(leer $GPUFREQ/cur_freq) / $(leer $GPUFREQ/min_freq) / $(leer $GPUFREQ/max_freq)"
    echo "gobernador GPU: $(leer $GPUFREQ/governor)"
    echo "nucleos CPU en linea: $(leer /sys/devices/system/cpu/online)"
    /usr/local/cuda/bin/nvcc --version | tail -1 || true
} | tee "$RES/sistema_gpu.txt"

echo "== 1. Reposo (5 s)"
con_tegrastats "$TEG/tegra_reposo.log" sleep 5
lineas=$( (grep -c POM_5V_IN "$TEG/tegra_reposo.log" 2> /dev/null) || true )
lineas=${lineas:-0}
echo "Muestras de reposo: $lineas"
if [ "$lineas" -lt 20 ]; then
    echo "ERROR: tegrastats registró muy pocas muestras; revisar antes de seguir." >&2
    exit 1
fi

echo "== 2. Tiempos instrumentados"
for m in "${MEDIDAS[@]}"; do
    rm -f "$RES/gpu_$m.csv"
    for p in "${PRECS[@]}"; do
        for n in "${NS[@]}"; do
            $BIN --precision "$p" --n "$n" --reps "$REPS_TIEMPO" --measure "$m" --csv "$RES/gpu_$m.csv" | tee -a "$SALIDA"
        done
    done
done

echo "== 3. Perfilador nvprof"
for m in "${MEDIDAS[@]}"; do
    for p in "${PRECS[@]}"; do
        for n in "${NS[@]}"; do
            sudo "$NVPROF" -u us --csv --log-file "$NVP/nvprof_${p}_${m}_n${n}.csv" \
                $BIN --precision "$p" --n "$n" --reps "$REPS_TIEMPO" --measure "$m" | tee -a "$SALIDA"
        done
    done
done

echo "== 4. Energía (unos 6 minutos)"
rm -f "$RES/gpu_energia.csv"
for m in "${MEDIDAS[@]}"; do
    for p in "${PRECS[@]}"; do
        for n in "${NS[@]}"; do
            con_tegrastats "$TEG/tegra_gpu_${p}_${m}_n${n}.log" \
                $BIN --precision "$p" --n "$n" --reps "$(reps_energia "$m" "$p" "$n")" --measure "$m" \
                     --csv "$RES/gpu_energia.csv"
        done
    done
done

echo "== 5. Tablas"
python3 nvprof_resumen.py --logs "$NVP" --salida "$RES/gpu_nvprof_resumen.csv" | tee -a "$SALIDA"
python3 energia.py --csv "$RES/gpu_energia.csv" --logs "$TEG" --reposo "$TEG/tegra_reposo.log" \
    --salida "$RES/gpu_energia_detalle.csv" --resumen "$RES/gpu_energia_resumen.csv" | tee -a "$SALIDA"

echo "Listo. Resultados en $RES/"
