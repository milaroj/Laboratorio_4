#!/usr/bin/env python3
"""nvprof_resumen.py — Laboratorio 4, Ejercicio B: resume los registros de nvprof.

Lee los archivos nvprof_<precision>_<medida>_n<N>.csv que genera medir_gpu.sh con
    sudo nvprof -u us --csv --log-file <archivo> ./softmax_cuda ...
y separa el tiempo de cómputo (kernel), las transferencias (memcpy HtoD/DtoH) y
el costo de las llamadas a la API (cudaLaunchKernel, cudaMemcpy). Todos los
tiempos están en microsegundos.
"""
import argparse
import csv
import glob
import os
import re
import sys

PATRON = re.compile(r'nvprof_(fp16|fp32)_(kernel|end-to-end)_n(\d+)\.csv$')


def leer(ruta):
    """Devuelve {(tipo, nombre): {'calls', 'avg', 'min', 'max', 'total'}}."""
    with open(ruta) as f:
        lineas = [l for l in f if not l.startswith('==') and l.strip()]
    filas = list(csv.reader(lineas))
    if not filas or filas[0][:1] != ['Type']:
        sys.exit('Formato inesperado en ' + ruta)
    enc = filas[0]
    i = {k: enc.index(k) for k in ('Type', 'Time', 'Calls', 'Avg', 'Min', 'Max', 'Name')}
    datos, tipo = {}, ''
    for r in filas[2:]:                       # filas[1] son las unidades
        if len(r) < len(enc):
            continue
        tipo = r[i['Type']] or tipo
        try:
            datos[(tipo, r[i['Name']])] = {
                'calls': int(r[i['Calls']]), 'avg': float(r[i['Avg']]), 'min': float(r[i['Min']]),
                'max': float(r[i['Max']]), 'total': float(r[i['Time']])}
        except ValueError:
            continue
    return datos


def buscar(datos, tipo, fragmento):
    for (t, nombre), v in datos.items():
        if t == tipo and fragmento in nombre:
            return v
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--logs', required=True, help='carpeta con los nvprof_*.csv')
    ap.add_argument('--salida', required=True, help='CSV de resumen')
    a = ap.parse_args()

    filas = []
    for ruta in glob.glob(os.path.join(a.logs, 'nvprof_*.csv')):
        m = PATRON.search(os.path.basename(ruta))
        if not m:
            continue
        d = leer(ruta)
        k = buscar(d, 'GPU activities', 'softmax_kernel')
        h2d = buscar(d, 'GPU activities', 'memcpy HtoD')
        d2h = buscar(d, 'GPU activities', 'memcpy DtoH')
        lanz = buscar(d, 'API calls', 'cudaLaunchKernel')
        cpy = buscar(d, 'API calls', 'cudaMemcpy')
        nan = float('nan')
        g = lambda v, c: v[c] if v else nan
        filas.append({
            'precision': m.group(1), 'medida': m.group(2), 'n': int(m.group(3)),
            'kernel_llamadas': g(k, 'calls'), 'kernel_avg_us': g(k, 'avg'),
            'kernel_min_us': g(k, 'min'), 'kernel_max_us': g(k, 'max'),
            'htod_llamadas': g(h2d, 'calls'), 'htod_avg_us': g(h2d, 'avg'),
            'dtoh_llamadas': g(d2h, 'calls'), 'dtoh_avg_us': g(d2h, 'avg'),
            'cudaLaunchKernel_avg_us': g(lanz, 'avg'), 'cudaMemcpy_avg_us': g(cpy, 'avg'),
        })
    if not filas:
        sys.exit('No se encontraron registros nvprof_*.csv en ' + a.logs)
    filas.sort(key=lambda f: (f['medida'] != 'kernel', f['precision'], f['n']))

    with open(a.salida, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print('nvprof (us)                   |  kernel GPU: prom    mín    máx | memcpy HtoD  DtoH | API: cudaLaunchKernel cudaMemcpy')
    for f in filas:
        print('%-4s %-10s N=%-5d         |  %14.3f %6.3f %6.3f | %11.3f %5.3f | %21.3f %10.3f'
              % (f['precision'], f['medida'], f['n'], f['kernel_avg_us'], f['kernel_min_us'],
                 f['kernel_max_us'], f['htod_avg_us'], f['dtoh_avg_us'],
                 f['cudaLaunchKernel_avg_us'], f['cudaMemcpy_avg_us']))
    print('En la medida "kernel" las copias se hacen una sola vez (entrada inicial y resultado final).')
    print('Resumen en ' + a.salida)


if __name__ == '__main__':
    main()
