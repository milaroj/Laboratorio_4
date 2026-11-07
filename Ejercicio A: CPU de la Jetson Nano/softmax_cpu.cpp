// softmax_cpu.cpp — Laboratorio 4, Ejercicio A (CPU de la Jetson Nano)
//
// Softmax numéricamente estable en FP16 y FP32:
//     m = max_i x_i,   y_i = exp(x_i - m) / sum_j exp(x_j - m)
// Referencia de exactitud: la misma función en double con std::exp.
//
// Decisión FP16: la CPU Cortex-A57 es ARMv8.0-A. Las instrucciones aritméticas
// en FP16 llegaron con ARMv8.2-A, así que en esta CPU __fp16 es solo un formato
// de almacenamiento: GCC convierte cada valor a float para operar. Además libm no
// ofrece exp en FP16. Por eso la variante FP16 guarda entrada y salida en FP16,
// pero calcula exp, el máximo y la suma en FP32.
//
// Compilación (Jetson):
//   g++ -O3 -std=c++17 -march=native softmax_cpu.cpp -o softmax_cpu -lm
// Uso:
//   ./softmax_cpu --precision fp16 --n 100 --reps 10000
//   ./softmax_cpu --precision fp32 --n 100 --reps 10000 --measure programa --csv cpu.csv

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------- tipo FP16
#if defined(__aarch64__) || defined(__arm__)
typedef __fp16 half_t;          // Jetson Nano (GCC en ARM)
#elif defined(__FLT16_MAX__)
typedef _Float16 half_t;        // x86 con GCC >= 12, solo para pruebas locales
#else
#error "El compilador no ofrece un tipo FP16 (__fp16 o _Float16)"
#endif

// ------------------------------------------------------------- configuración
struct Config {
    std::string precision = "fp32";   // fp16 | fp32
    std::string medida = "funcion";   // funcion | programa
    int n = 100;
    long reps = 10000;
    int corridas = 5;                 // mediciones válidas (sin contar calentamiento)
    uint32_t semilla = 42;
    std::string csv;                  // archivo CSV para datos crudos (opcional)
    std::string guardar_entrada;      // guarda el vector de entrada (opcional)
};

static void uso(const char* prog) {
    std::printf(
        "Uso: %s --precision fp16|fp32 --n N --reps R [opciones]\n"
        "  --measure funcion|programa  qué se mide (por defecto: funcion)\n"
        "  --runs K                    mediciones tras el calentamiento (por defecto: 5)\n"
        "  --seed S                    semilla de la entrada (por defecto: 42)\n"
        "  --csv archivo.csv           agrega una fila por medición\n"
        "  --guardar-entrada arch.csv  guarda el vector de entrada usado\n", prog);
}

static Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto valor = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "Falta valor para %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if      (a == "--precision")        c.precision = valor();
        else if (a == "--n")                c.n = std::atoi(valor().c_str());
        else if (a == "--reps")             c.reps = std::atol(valor().c_str());
        else if (a == "--runs")             c.corridas = std::atoi(valor().c_str());
        else if (a == "--seed")             c.semilla = (uint32_t)std::strtoul(valor().c_str(), nullptr, 10);
        else if (a == "--measure")          c.medida = valor();
        else if (a == "--csv")              c.csv = valor();
        else if (a == "--guardar-entrada")  c.guardar_entrada = valor();
        else if (a == "-h" || a == "--help") { uso(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "Opción desconocida: %s\n", a.c_str()); uso(argv[0]); std::exit(1); }
    }
    if ((c.precision != "fp16" && c.precision != "fp32") ||
        (c.medida != "funcion" && c.medida != "programa") ||
        c.n < 1 || c.reps < 1 || c.corridas < 1) {
        uso(argv[0]);
        std::exit(1);
    }
    return c;
}

// ------------------------------------------------------------------ entrada
// Mismo generador en CPU, GPU y KV260: mt19937 con la semilla dada y un mapeo
// explícito a [-4, 4). No se usa uniform_real_distribution porque su algoritmo
// depende de la biblioteca estándar.
static void generar_entrada(std::vector<float>& x, uint32_t semilla) {
    std::mt19937 gen(semilla);
    for (size_t i = 0; i < x.size(); ++i) {
        double u = gen() / 4294967296.0;            // [0, 1)
        x[i] = (float)(-4.0 + 8.0 * u);
    }
}

// --------------------------------------------------------------- softmax
// Referencia FP64.
static void softmax_ref(const std::vector<float>& x, std::vector<double>& y) {
    double m = x[0];
    for (float v : x) m = std::max(m, (double)v);
    double s = 0.0;
    for (size_t i = 0; i < x.size(); ++i) { y[i] = std::exp((double)x[i] - m); s += y[i]; }
    for (double& v : y) v /= s;
}

// Versión medida. T es float (FP32) o half_t (FP16). Las operaciones se hacen
// en float; con T = half_t se convierte al leer y al escribir. El exp de cada
// elemento se guarda en y (en formato T) y la última pasada lo normaliza.
template <typename T>
__attribute__((noinline))
void softmax(const T* __restrict x, T* __restrict y, int n) {
    float m = (float)x[0];
    for (int i = 1; i < n; ++i) m = std::max(m, (float)x[i]);   // 1) máximo

    float s = 0.0f;
    for (int i = 0; i < n; ++i) {                                 // 2) exp y suma
        float e = std::exp((float)x[i] - m);
        y[i] = (T)e;
        s += e;
    }

    for (int i = 0; i < n; ++i) y[i] = (T)((float)y[i] / s);      // 3) normalización
}

// ---------------------------------------------------------------- utilidades
static double ahora_s() {               // reloj monotónico para medir intervalos
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
static double epoch_s() {               // reloj de pared para alinear con tegrastats
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

static std::string leer_linea(const std::string& ruta) {
    std::ifstream f(ruta);
    std::string s;
    if (f) std::getline(f, s);
    return s;
}

static long frecuencia_cpu_khz(int cpu) {
    std::string s = leer_linea("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/scaling_cur_freq");
    return s.empty() ? -1 : std::atol(s.c_str());
}

static double temperatura_cpu_c() {     // zona térmica cuyo tipo contiene "CPU"
    for (int z = 0; z < 16; ++z) {
        std::string base = "/sys/devices/virtual/thermal/thermal_zone" + std::to_string(z);
        std::string tipo = leer_linea(base + "/type");
        if (tipo.empty()) continue;
        if (tipo.find("CPU") != std::string::npos || tipo.find("cpu") != std::string::npos) {
            std::string t = leer_linea(base + "/temp");
            if (!t.empty()) return std::atof(t.c_str()) / 1000.0;
        }
    }
    return NAN;
}

static void imprimir_sistema() {
    std::string online = leer_linea("/sys/devices/system/cpu/online");
    std::string gob = leer_linea("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    std::printf("Sistema: núcleos en línea %s | gobernador %s | frecuencias (MHz):",
                online.empty() ? "?" : online.c_str(), gob.empty() ? "?" : gob.c_str());
    for (int c = 0; c < 8; ++c) {
        long f = frecuencia_cpu_khz(c);
        if (f > 0) std::printf(" cpu%d=%ld", c, f / 1000);
    }
    std::printf(" | temp CPU %.1f C\n", temperatura_cpu_c());
}

struct Verificacion {
    double suma, err_abs_max, err_abs_medio, err_rel_max, err_rel_medio;
    bool rango_ok, suma_ok, argmax_ok, empate;
    int idx_ref, idx_obt;
};

template <typename T>
static Verificacion verificar(const std::vector<T>& y, const std::vector<double>& ref, double tol_suma) {
    Verificacion v{};
    int n = (int)y.size();
    v.rango_ok = true;
    for (int i = 0; i < n; ++i) {
        double yi = (double)(float)y[i];
        double ea = std::fabs(yi - ref[i]);
        double er = ea / ref[i];                     // ref[i] > 0 siempre en softmax
        v.suma += yi;
        v.err_abs_max = std::max(v.err_abs_max, ea);
        v.err_rel_max = std::max(v.err_rel_max, er);
        v.err_abs_medio += ea;
        v.err_rel_medio += er;
        if (!(yi >= 0.0 && yi <= 1.0)) v.rango_ok = false;
    }
    v.err_abs_medio /= n;
    v.err_rel_medio /= n;
    v.suma_ok = std::fabs(v.suma - 1.0) <= tol_suma;
    v.idx_ref = (int)(std::max_element(ref.begin(), ref.end()) - ref.begin());
    v.idx_obt = 0;
    for (int i = 1; i < n; ++i) if ((float)y[i] > (float)y[v.idx_obt]) v.idx_obt = i;
    // En FP16 dos valores cercanos pueden redondear al mismo número; si la
    // posición de referencia también tiene el valor máximo se considera empate.
    v.empate = (v.idx_obt != v.idx_ref) && ((float)y[v.idx_ref] == (float)y[v.idx_obt]);
    v.argmax_ok = (v.idx_obt == v.idx_ref) || v.empate;
    return v;
}

static double mediana(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t k = v.size();
    return (k % 2) ? v[k / 2] : 0.5 * (v[k / 2 - 1] + v[k / 2]);
}

// ---------------------------------------------------------------- ejecución
template <typename T>
static int ejecutar(const Config& c) {
    const int n = c.n;
    const double tol_suma = (c.precision == "fp16") ? 5e-3 : 1e-5;

    std::printf("=== softmax_cpu | %s | N=%d | reps=%ld | medida=%s | semilla=%u ===\n",
                c.precision.c_str(), n, c.reps, c.medida.c_str(), c.semilla);
    imprimir_sistema();

    // Entrada y referencia (fuera de la medición en modo "funcion").
    std::vector<float> xf(n);
    generar_entrada(xf, c.semilla);
    std::vector<double> ref(n);
    softmax_ref(xf, ref);
    std::vector<T> x(n), y(n);
    for (int i = 0; i < n; ++i) x[i] = (T)xf[i];

    if (!c.guardar_entrada.empty()) {
        std::ofstream f(c.guardar_entrada);
        f << "i,x\n";
        char buf[64];
        for (int i = 0; i < n; ++i) { std::snprintf(buf, sizeof buf, "%d,%.9g\n", i, xf[i]); f << buf; }
        std::printf("Entrada guardada en %s\n", c.guardar_entrada.c_str());
    }

    struct Fila { double total, por_llamada_us, t_ini, t_fin, temp; long frec; };
    std::vector<Fila> filas;

    for (int r = 0; r <= c.corridas; ++r) {        // r = 0 es el calentamiento
        double t_ini = epoch_s();
        double t0 = ahora_s();

        if (c.medida == "funcion") {
            for (long k = 0; k < c.reps; ++k) {
                softmax<T>(x.data(), y.data(), n);
                asm volatile("" : : "r"(y.data()) : "memory");   // evita que se omitan llamadas
            }
        } else {
            // Programa completo: reservar, generar entrada, convertir, calcular,
            // obtener la referencia y verificar.
            std::vector<float> xf2(n);
            generar_entrada(xf2, c.semilla);
            std::vector<T> x2(n), y2(n);
            for (int i = 0; i < n; ++i) x2[i] = (T)xf2[i];
            for (long k = 0; k < c.reps; ++k) {
                softmax<T>(x2.data(), y2.data(), n);
                asm volatile("" : : "r"(y2.data()) : "memory");
            }
            std::vector<double> ref2(n);
            softmax_ref(xf2, ref2);
            Verificacion v2 = verificar(y2, ref2, tol_suma);
            asm volatile("" : : "r"(&v2) : "memory");
            y = y2;
        }

        double t1 = ahora_s();
        double t_fin = epoch_s();
        double total = t1 - t0;
        Fila f{total, total / c.reps * 1e6, t_ini, t_fin, temperatura_cpu_c(), frecuencia_cpu_khz(0)};
        if (r == 0) {
            std::printf("Calentamiento: %.3f ms total, %.3f us/llamada\n", total * 1e3, f.por_llamada_us);
        } else {
            std::printf("Medición %d: %.3f ms total, %.4f us/llamada\n", r, total * 1e3, f.por_llamada_us);
            filas.push_back(f);
        }
    }

    // Resumen de tiempos
    std::vector<double> t;
    for (auto& f : filas) t.push_back(f.por_llamada_us);
    std::printf("Tiempo por llamada (us): mediana %.4f | mín %.4f | máx %.4f\n",
                mediana(t), *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));

    // Verificación contra FP64
    Verificacion v = verificar(y, ref, tol_suma);
    bool ok = v.rango_ok && v.suma_ok && v.argmax_ok;
    std::printf("Verificación contra FP64:\n");
    std::printf("  suma de la salida = %.7f (|suma-1| = %.2e, tolerancia %.0e) %s\n",
                v.suma, std::fabs(v.suma - 1.0), tol_suma, v.suma_ok ? "PASA" : "FALLA");
    std::printf("  todos en [0,1]: %s\n", v.rango_ok ? "PASA" : "FALLA");
    std::printf("  posición del máximo: referencia %d, obtenida %d %s\n",
                v.idx_ref, v.idx_obt, v.argmax_ok ? (v.empate ? "PASA (empate en FP16)" : "PASA") : "FALLA");
    std::printf("  error absoluto: máx %.3e | medio %.3e\n", v.err_abs_max, v.err_abs_medio);
    std::printf("  error relativo: máx %.3e | medio %.3e\n", v.err_rel_max, v.err_rel_medio);
    std::printf("Resultado: %s\n", ok ? "CORRECTO" : "INCORRECTO");

    // Datos crudos en CSV (una fila por medición válida)
    if (!c.csv.empty()) {
        bool nuevo = leer_linea(c.csv).empty();
        std::ofstream f(c.csv, std::ios::app);
        if (nuevo)
            f << "plataforma,precision,medida,n,reps,medicion,tiempo_total_s,tiempo_llamada_us,"
                 "t_inicio_epoch,t_fin_epoch,frec_cpu0_khz,temp_cpu_c,suma,err_abs_max,err_abs_medio,"
                 "err_rel_max,err_rel_medio,verificacion\n";
        char buf[512];
        for (size_t i = 0; i < filas.size(); ++i) {
            const Fila& r = filas[i];
            std::snprintf(buf, sizeof buf,
                          "cpu,%s,%s,%d,%ld,%zu,%.9f,%.6f,%.3f,%.3f,%ld,%.1f,%.9f,%.6e,%.6e,%.6e,%.6e,%s\n",
                          c.precision.c_str(), c.medida.c_str(), n, c.reps, i + 1, r.total, r.por_llamada_us,
                          r.t_ini, r.t_fin, r.frec, r.temp, v.suma, v.err_abs_max, v.err_abs_medio,
                          v.err_rel_max, v.err_rel_medio, ok ? "correcto" : "incorrecto");
            f << buf;
        }
        std::printf("CSV: %zu filas agregadas a %s\n", filas.size(), c.csv.c_str());
    }
    return ok ? 0 : 2;
}

int main(int argc, char** argv) {
    Config c = parse_args(argc, argv);
    return (c.precision == "fp16") ? ejecutar<half_t>(c) : ejecutar<float>(c);
}
