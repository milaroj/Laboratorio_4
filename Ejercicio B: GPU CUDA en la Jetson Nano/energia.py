#!/usr/bin/env python3
"""energia.py — Laboratorio 4: cruza las mediciones con los registros de tegrastats.

(Copia del Ejercicio A; aquí el resumen también muestra GPU: energía del riel,
utilización y frecuencia de GR3D y temperatura.)

Para cada fila del CSV de mediciones busca su registro de tegrastats
(tegra_<plataforma>_<precision>_<medida>_n<N>.log), toma las muestras que caen
dentro de [t_inicio + margen, t_fin - margen] y calcula la potencia promedio
de cada riel y la energía por llamada.

Rieles del monitor INA3221 de la Jetson Nano, según tegrastats:
  POM_5V_IN   entrada de 5 V del módulo: módulo completo (CPU, GPU, memoria...)
  POM_5V_CPU  riel de la CPU
  POM_5V_GPU  riel de la GPU (el acelerador)

tegrastats no imprime la hora de cada muestra. medir_*.sh guarda en <log>.t0
la hora de inicio, la hora de fin y el intervalo; aquí la muestra k se ubica
en t0 + (k + 0.5) * dt, con dt = (t1 - t0) / muestras (corrige la deriva).

Unidades: potencia en mW, tiempo por llamada en us; mW * us = nJ, /1000 = uJ.
"""
import argparse
import csv
import math
import os
import re
import statistics
import sys

RE_POM = {r: re.compile(r'POM_5V_%s (\d+)/' % r) for r in ('IN', 'CPU', 'GPU')}
RE_NUCLEOS = re.compile(r'CPU \[([^\]]*)\]')
RE_TCPU = re.compile(r'CPU@([\d.]+)C')
RE_TGPU = re.compile(r'GPU@([\d.]+)C')
RE_GR3D = re.compile(r'GR3D_FREQ (\d+)%@(\d+)')
NAN = float('nan')


def leer_log(ruta):
    with open(ruta + '.t0') as f:
        t0, t1, _ = f.read().split()
    t0, t1 = float(t0), float(t1)
    with open(ruta) as f:
        lineas = [l for l in f if 'POM_5V_IN' in l]
    if not lineas:
        sys.exit('Sin muestras de potencia en ' + ruta)
    dt = (t1 - t0) / len(lineas)
    muestras = []
    for k, l in enumerate(lineas):
        m = {'t': t0 + (k + 0.5) * dt}
        for r, rx in RE_POM.items():
            mm = rx.search(l)
            m['p_' + r.lower()] = float(mm.group(1)) if mm else NAN
        usos, frecs = [], []
        mm = RE_NUCLEOS.search(l)
        if mm:
            for c in mm.group(1).split(','):
                if '@' in c:                       # los núcleos apagados dicen "off"
                    u, f = c.split('@')
                    usos.append(float(u.rstrip('%')))
                    frecs.append(float(f))
        if usos:
            i = usos.index(max(usos))
            m['uso_max'], m['frec_ocupado'] = usos[i], frecs[i]
        else:
            m['uso_max'] = m['frec_ocupado'] = NAN
        m['nucleos_activos'] = len(usos)
        mm = RE_TCPU.search(l)
        m['temp_cpu'] = float(mm.group(1)) if mm else NAN
        mm = RE_TGPU.search(l)
        m['temp_gpu'] = float(mm.group(1)) if mm else NAN
        mm = RE_GR3D.search(l)
        m['gr3d_pct'], m['gr3d_mhz'] = (float(mm.group(1)), float(mm.group(2))) if mm else (NAN, NAN)
        muestras.append(m)
    return muestras


def promedio(muestras, clave):
    v = [m[clave] for m in muestras if not math.isnan(m[clave])]
    return sum(v) / len(v) if v else NAN


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', required=True, help='CSV de mediciones (con t_inicio_epoch y t_fin_epoch)')
    ap.add_argument('--logs', required=True, help='carpeta con los tegra_*.log')
    ap.add_argument('--reposo', required=True, help='registro de tegrastats en reposo')
    ap.add_argument('--salida', required=True, help='CSV con una fila por medición')
    ap.add_argument('--resumen', required=True, help='CSV con mediana/mín/máx por configuración')
    ap.add_argument('--margen', type=float, default=0.2, help='segundos que se descartan en cada borde')
    a = ap.parse_args()

    reposo = leer_log(a.reposo)[2:]               # descarta las primeras muestras
    p_rep = {r: promedio(reposo, 'p_' + r) for r in ('in', 'cpu', 'gpu')}
    print('Reposo: POM_5V_IN %.0f mW | POM_5V_CPU %.0f mW | POM_5V_GPU %.0f mW (%d muestras)'
          % (p_rep['in'], p_rep['cpu'], p_rep['gpu'], len(reposo)))

    with open(a.csv) as f:
        filas = list(csv.DictReader(f))
    cache, salida = {}, []
    for r in filas:
        nombre = 'tegra_%s_%s_%s_n%s.log' % (r['plataforma'], r['precision'], r['medida'], r['n'])
        ruta = os.path.join(a.logs, nombre)
        if ruta not in cache:
            cache[ruta] = leer_log(ruta)
        t_ini, t_fin = float(r['t_inicio_epoch']), float(r['t_fin_epoch'])
        sel = [m for m in cache[ruta] if t_ini + a.margen <= m['t'] <= t_fin - a.margen]
        if len(sel) < 5:
            print('AVISO: %s medición %s tiene solo %d muestras; aumentar --reps'
                  % (nombre, r['medicion'], len(sel)))
        if not sel:
            continue
        t_us = float(r['tiempo_llamada_us'])
        p = {k: promedio(sel, 'p_' + k) for k in ('in', 'cpu', 'gpu')}
        salida.append({
            'plataforma': r['plataforma'], 'precision': r['precision'], 'medida': r['medida'],
            'n': int(r['n']), 'reps': r['reps'], 'medicion': r['medicion'],
            'tiempo_llamada_us': t_us, 'muestras': len(sel),
            'p_in_mw': p['in'], 'p_cpu_mw': p['cpu'], 'p_gpu_mw': p['gpu'],
            'p_in_reposo_mw': p_rep['in'], 'p_cpu_reposo_mw': p_rep['cpu'], 'p_gpu_reposo_mw': p_rep['gpu'],
            'e_in_uj': p['in'] * t_us / 1000,
            'e_in_dinamica_uj': (p['in'] - p_rep['in']) * t_us / 1000,
            'e_cpu_uj': p['cpu'] * t_us / 1000,
            'e_gpu_uj': p['gpu'] * t_us / 1000,
            'frec_nucleo_ocupado_mhz': promedio(sel, 'frec_ocupado'),
            'uso_max_pct': promedio(sel, 'uso_max'),
            'nucleos_activos': max(m['nucleos_activos'] for m in sel),
            'temp_cpu_c': max(m['temp_cpu'] for m in sel),
            'temp_gpu_c': max(m['temp_gpu'] for m in sel),
            'gr3d_pct': promedio(sel, 'gr3d_pct'), 'gr3d_mhz': promedio(sel, 'gr3d_mhz'),
        })

    if not salida:
        sys.exit('No hubo mediciones con muestras de potencia.')
    with open(a.salida, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(salida[0].keys()))
        w.writeheader()
        w.writerows(salida)

    # Resumen por configuración: mediana, mínimo y máximo
    campos = ['tiempo_llamada_us', 'p_in_mw', 'p_cpu_mw', 'p_gpu_mw', 'e_in_uj', 'e_in_dinamica_uj',
              'e_cpu_uj', 'e_gpu_uj', 'frec_nucleo_ocupado_mhz', 'gr3d_pct', 'gr3d_mhz',
              'temp_cpu_c', 'temp_gpu_c']
    grupos = {}
    for s in salida:
        grupos.setdefault((s['plataforma'], s['precision'], s['medida'], s['n']), []).append(s)
    resumen = []
    for clave in sorted(grupos):
        g = grupos[clave]
        fila = dict(zip(('plataforma', 'precision', 'medida', 'n'), clave))
        fila['mediciones'] = len(g)
        for c in campos:
            v = [x[c] for x in g]
            fila[c + '_mediana'] = statistics.median(v)
            fila[c + '_min'] = min(v)
            fila[c + '_max'] = max(v)
        resumen.append(fila)
    with open(a.resumen, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)

    gpu = any(f['plataforma'] == 'gpu' for f in resumen)
    if gpu:
        cab = ('E_GPU', 'GR3D', 'MHz', 'T GPU')
        und = ('(uJ)', '(%)', 'GPU', 'máx')
    else:
        cab = ('E_CPU', 'uso', 'MHz', 'T CPU')
        und = ('(uJ)', '', 'CPU', 'máx')
    fmt_c = '%-6s %-10s %5s | %10s | %8s %8s %8s | %9s %9s | %9s %5s %5s %6s'
    print(fmt_c % (('prec', 'medida', 'N', 't (us)', 'P_IN', 'P_CPU', 'P_GPU', 'E_IN', 'E_IN-rep') + cab))
    print(fmt_c % (('', '', '', 'mediana', '(mW)', '(mW)', '(mW)', '(uJ)', '(uJ)') + und))
    for f in resumen:
        if gpu:
            extra = (f['e_gpu_uj_mediana'], f['gr3d_pct_mediana'], f['gr3d_mhz_mediana'], f['temp_gpu_c_max'])
        else:
            extra = (f['e_cpu_uj_mediana'], 100.0, f['frec_nucleo_ocupado_mhz_mediana'], f['temp_cpu_c_max'])
        print('%-6s %-10s %5d | %10.4f | %8.0f %8.0f %8.0f | %9.4f %9.4f | %9.4f %5.0f %5.0f %6.1f'
              % ((f['precision'], f['medida'], f['n'], f['tiempo_llamada_us_mediana'],
                  f['p_in_mw_mediana'], f['p_cpu_mw_mediana'], f['p_gpu_mw_mediana'],
                  f['e_in_uj_mediana'], f['e_in_dinamica_uj_mediana']) + extra))
    print('Medianas de %d mediciones por configuración. Detalle en %s; mín/máx en %s'
          % (min(len(g) for g in grupos.values()), a.salida, a.resumen))


if __name__ == '__main__':
    main()
