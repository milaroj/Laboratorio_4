#!/bin/bash
# medir_cpu.sh — Laboratorio 4, Ejercicio A: corre todas las mediciones de la CPU.
#
#  1. Reposo: registra tegrastats 5 s sin carga (potencia de referencia).
#  2. Tiempos de la función:   N en {10,100,500,1000}, FP16 y FP32, 10 000 reps.
#  3. Tiempos del programa completo: mismas configuraciones, 10 000 reps.
#  4. Energía: la función con las repeticiones necesarias para que cada medición
#     dure ~3 s, mientras tegrastats muestrea cada 100 ms.
#  5. energia.py cruza cada medición con sus muestras de potencia.
#
# Uso, en la Jetson y desde la carpeta del ejercicio:  bash medir_cpu.sh
# Pide la contraseña de sudo una vez (tegrastats necesita sudo para la potencia).
set -euo pipefail

BIN=./softmax_cpu
RES="$PWD/resultados"
TEG=$RES/tegrastats
INTERVALO=100                    # ms entre muestras de tegrastats
NS=(10 100 500 1000)
PRECS=(fp16 fp32)
REPS_TIEMPO=10000
DURACION=3                       # segundos por medición en las corridas de energía

# Repeticiones para que una medición dure ~DURACION s, según el tiempo por
# llamada medido en el paso 2 (columna 8 de cpu_funcion.csv).
reps_energia() {
    awk -F, -v p="$1" -v n="$2" -v d="$DURACION" \
        'NR > 1 && $2 == p && $4 == n { s += $8; c++ } END { printf "%d\n", d * 1e6 / (s / c) }' \
        "$RES/cpu_funcion.csv"
}

mkdir -p "$TEG"
SALIDA=$RES/salida_cpu.txt
: > "$SALIDA"
sudo -v

# Graba tegrastats mientras corre un comando.
# Guarda en <log>.t0: hora de inicio, hora de fin e intervalo, para ubicar cada muestra.
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

echo "== Estado del sistema"
{
    echo "fecha: $(date -Is)"
    echo "host: $(hostname)"
    sudo nvpmodel -q 2>/dev/null | grep -v WARN
    echo "nucleos en linea: $(cat /sys/devices/system/cpu/online)"
    echo "gobernador: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "frecuencia max CPU (kHz): $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
    g++ --version | head -1
} | tee "$RES/sistema_cpu.txt"

echo "== 1. Reposo (5 s)"
con_tegrastats "$TEG/tegra_reposo.log" sleep 5
lineas=$( (grep -c POM_5V_IN "$TEG/tegra_reposo.log" 2> /dev/null) || true )
lineas=${lineas:-0}
echo "Muestras de reposo: $lineas"
if [ "$lineas" -lt 20 ]; then
    echo "ERROR: tegrastats registró muy pocas muestras; revisar antes de seguir." >&2
    exit 1
fi

echo "== 2. Tiempos de la función"
rm -f "$RES/cpu_funcion.csv"
for p in "${PRECS[@]}"; do
    for n in "${NS[@]}"; do
        $BIN --precision "$p" --n "$n" --reps "$REPS_TIEMPO" --csv "$RES/cpu_funcion.csv" | tee -a "$SALIDA"
    done
done

echo "== 3. Tiempos del programa completo"
rm -f "$RES/cpu_programa.csv"
for p in "${PRECS[@]}"; do
    for n in "${NS[@]}"; do
        $BIN --precision "$p" --n "$n" --reps "$REPS_TIEMPO" --measure programa --csv "$RES/cpu_programa.csv" | tee -a "$SALIDA"
    done
done

echo "== 4. Energía (unos 3 minutos)"
rm -f "$RES/cpu_energia.csv"
for p in "${PRECS[@]}"; do
    for n in "${NS[@]}"; do
        con_tegrastats "$TEG/tegra_cpu_${p}_funcion_n${n}.log" \
            $BIN --precision "$p" --n "$n" --reps "$(reps_energia "$p" "$n")" --csv "$RES/cpu_energia.csv"
    done
done

echo "== 5. Potencia y energía por medición"
python3 energia.py --csv "$RES/cpu_energia.csv" --logs "$TEG" --reposo "$TEG/tegra_reposo.log" \
    --salida "$RES/cpu_energia_detalle.csv" --resumen "$RES/cpu_energia_resumen.csv" | tee -a "$SALIDA"

echo "Listo. Resultados en $RES/"
