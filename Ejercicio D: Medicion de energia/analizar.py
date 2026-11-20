"""Calcula la comparación entre plataformas y genera las gráficas del inciso D."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
CPU = BASE.parent / "Ejercicio A: CPU de la Jetson Nano" / "resultados"
GPU = BASE.parent / "Ejercicio B: GPU CUDA en la Jetson Nano" / "resultados"
KRIA_FP32 = BASE.parent / "Ejercicio C: HLS para KV260" / "SW"
OUT = BASE / "resultados"
MODOS = ("CPU función", "GPU kernel (host)", "GPU extremo a extremo", "KV260 host")
TAMANOS = (10, 100, 500, 1000)


def leer(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def agrupar(filas):
    grupos = defaultdict(list)
    for fila in filas:
        grupos[(fila["precision"].upper(), int(fila["n"]))].append(fila)
    return grupos


def resumen(clave, modo, variante, tiempos, potencias, energias, errores, nvprof=None):
    precision, n = clave
    fila = {
        "precision": precision,
        "n": n,
        "modo": modo,
        "variante": variante,
        "lotes": len(energias),
        "tiempo_us": mean(tiempos),
        "de_tiempo_us": stdev(tiempos) if len(tiempos) > 1 else None,
        "potencia_w": mean(potencias),
        "energia_uj": mean(energias),
        "de_energia_uj": stdev(energias) if len(energias) > 1 else None,
        "energia_j_elemento": mean(energias) * 1e-6 / n,
        "de_energia_j_elemento": (
            stdev(energias) * 1e-6 / n if len(energias) > 1 else None
        ),
        "error_abs_max": max(errores),
        "error_abs_max_media": mean(errores),
        "de_error_abs_max": stdev(errores) if len(errores) > 1 else None,
        "kernel_nvprof_us": nvprof,
    }

    for nombre, valores in (
        ("tiempo_us", tiempos),
        ("potencia_w", potencias),
        ("energia_uj", energias),
        ("energia_j_elemento", [e * 1e-6 / n for e in energias]),
        ("error_abs_max", errores),
    ):
        fila[f"{nombre}_mediana"] = median(valores)
        fila[f"{nombre}_min"] = min(valores) if len(valores) > 1 else None
        fila[f"{nombre}_max"] = max(valores) if len(valores) > 1 else None

    return fila


def analizar():
    energia_cpu = leer(CPU / "cpu_energia_detalle.csv")
    energia_gpu = leer(GPU / "gpu_energia_detalle.csv")
    perfil = {
        (r["precision"].upper(), int(r["n"])): float(r["kernel_avg_us"])
        for r in leer(GPU / "gpu_nvprof_resumen.csv")
        if r["medida"] == "end-to-end"
    }
    jetson = (
        (MODOS[0], "funcion", energia_cpu, CPU / "cpu_funcion.csv"),
        (MODOS[1], "kernel", energia_gpu, GPU / "gpu_kernel.csv"),
        (MODOS[2], "end-to-end", energia_gpu, GPU / "gpu_end-to-end.csv"),
    )

    resultados = []
    for modo, medida, energia, validacion in jetson:
        lotes = agrupar(r for r in energia if r["medida"] == medida)
        errores = agrupar(leer(validacion))

        for clave, filas in lotes.items():
            assert len(filas) == len(errores[clave]) == 5, (modo, clave)
            resultados.append(resumen(
                clave,
                modo,
                "",
                [float(r["tiempo_llamada_us"]) for r in filas],
                [float(r["p_in_mw"]) / 1000 for r in filas],
                [float(r["e_in_uj"]) for r in filas],
                [float(r["err_abs_max"]) for r in errores[clave]],
                perfil[clave] if medida == "kernel" else None,
            ))

    crudos_fp16 = leer(OUT / "energia_kv260_fp16.csv")
    assert len(crudos_fp16) == 48
    assert len({(r["variante"], r["n"], r["lote"]) for r in crudos_fp16}) == 48

    # Corridas de FP32 V0 guardadas en SW/resultados_<N>/.
    crudos_fp32 = [
        fila
        for archivo in KRIA_FP32.glob(
            "resultados_*/resultados_softmax_fp32_V0_*_resumen.csv"
        )
        for fila in leer(archivo)
    ]

    # Lotes 4 y 5 de FP16 V3 para cada tamaño de entrada.
    nuevos_v3 = [
        fila
        for archivo in sorted(
            (OUT / "complemento_V3.BNoUbZ").glob("N*/lote_*/*_resumen.csv")
        )
        for fila in leer(archivo)
    ]
    assert len(nuevos_v3) == 8
    assert all(
        fila["precision"] == "FP16" and fila["variante"] == "V3"
        for fila in nuevos_v3
    )

    kria = [r for r in crudos_fp16 if r["variante"] == "V3"] + nuevos_v3 + crudos_fp32

    for clave, filas in agrupar(kria).items():
        assert len(filas) == (5 if clave[0] == "FP16" else 1)
        assert all(r["estado"] == "PASS" for r in filas)

        resultados.append(resumen(
            clave,
            MODOS[3],
            filas[0]["variante"],
            [float(r["total_us_por_llamada_amortizado"]) for r in filas],
            [float(r["potencia_promedio_w"]) for r in filas],
            [float(r["energia_por_llamada_uj"]) for r in filas],
            [float(r["error_max"]) for r in filas],
        ))

    assert len(resultados) == 32
    return sorted(resultados, key=lambda r: (r["precision"], r["n"], MODOS.index(r["modo"])))


def graficar(filas):
    metricas = (
        ("Tiempo por llamada (µs)", "tiempo_us", "de_tiempo_us"),
        ("Energía por llamada (µJ)", "energia_uj", "de_energia_uj"),
        ("Energía por elemento (J)", "energia_j_elemento", "de_energia_j_elemento"),
        ("Error absoluto máximo frente a FP64", "error_abs_max_media", "de_error_abs_max"),
    )
    colores = ("#0b6964", "#2454a2", "#a84b1d", "#793e8b")
    marcas = ("o", "^", "s", "D")
    fig, ejes = plt.subplots(4, 2, figsize=(10, 12), sharex=True)

    for i, (titulo, y, de) in enumerate(metricas):
        for j, precision in enumerate(("FP16", "FP32")):
            ax = ejes[i, j]

            for modo, color, marca in zip(MODOS, colores, marcas):
                datos = [
                    r for r in filas
                    if r["precision"] == precision and r["modo"] == modo
                ]
                ax.plot(
                    [r["n"] for r in datos],
                    [r[y] for r in datos],
                    color=color,
                    marker=marca,
                    markersize=5,
                    label=modo,
                )
                for r in datos:
                    if r[de]:
                        ax.errorbar(
                            r["n"], r[y], yerr=r[de],
                            color=color, fmt="none", capsize=3,
                        )

            ax.set(xscale="log", yscale="log", xlim=(8, 1250))
            ax.set_xticks(TAMANOS, [str(n) for n in TAMANOS])
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)

            if i == 0:
                ax.set_title(precision, fontweight="bold")
            if j == 0:
                ax.set_ylabel(titulo)
            if i == 3:
                ax.set_xlabel("Elementos del vector (N)")

        limites = [ax.get_ylim() for ax in ejes[i]]
        for ax in ejes[i]:
            ax.set_ylim(min(a for a, _ in limites), max(b for _, b in limites))

    fig.suptitle("Softmax · Tiempo, energía y exactitud", fontsize=15, fontweight="bold")
    fig.legend(
        *ejes[0, 0].get_legend_handles_labels(),
        ncol=2,
        loc="lower center",
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    fig.savefig(OUT / "graficas_softmax.png", dpi=160)
    plt.close(fig)


def main():
    filas = analizar()
    with (OUT / "comparacion.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=filas[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(filas)

    graficar(filas)


if __name__ == "__main__":
    main()