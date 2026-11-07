// softmax_cuda.cu — Laboratorio 4, Ejercicio B (GPU CUDA de la Jetson Nano)
//
// Softmax numéricamente estable en FP16 (__half) y FP32 (float):
//     m = max_i x_i,   y_i = exp(x_i - m) / sum_j exp(x_j - m)
// Referencia de exactitud: la misma función en double con std::exp (en la CPU).
//
// Organización (N <= 1000 cabe en un solo bloque):
//   - 1 bloque; hilos = N redondeado a múltiplo de 32, máximo 256 (--block).
//     Cada hilo recorre los elementos i = tid, tid + hilos, ... (acceso coalescido).
//   - Las tres etapas van en un solo kernel, separadas por reducciones de bloque:
//       1) máximo:    máximo local por hilo -> reducción en el warp con
//                     __shfl_down_sync -> reducción de los warps en memoria compartida.
//       2) exp/suma:  e_i = exp(x_i - m) se guarda en y (memoria global); la suma
//                     local se reduce igual que el máximo.
//       3) normalización: y_i = e_i / suma.
//   - Memoria: x e y en memoria global; 32 floats en memoria compartida para los
//     resultados parciales de cada warp.
//
// Decisión FP16: la GPU Maxwell (sm_53) sí tiene aritmética FP16 nativa, así que la
// resta, el exp (hexp) y la división (__hdiv) se hacen en __half. Dos excepciones:
//   - El máximo se compara convirtiendo a float: es exacto, el resultado es el
//     mismo valor __half.
//   - La suma se acumula en float. Una suma en __half de hasta 1000 términos
//     pierde precisión (el espaciado de __half cerca de 1 es 9.8e-4); al final se
//     redondea a __half para la división.
//
// Compilación (Jetson, CUDA 10.2). -arch=sm_53 es necesario: con la arquitectura
// por defecto de nvcc 10.2 (sm_30) no existen las operaciones aritméticas __half.
//   nvcc -O3 -std=c++14 -arch=sm_53 softmax_cuda.cu -o softmax_cuda
// Uso:
//   ./softmax_cuda --precision fp16 --n 100 --reps 10000 --measure kernel
//   ./softmax_cuda --precision fp16 --n 100 --reps 10000 --measure end-to-end

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define CK(llamada)                                                                   \
    do {                                                                              \
        cudaError_t e_ = (llamada);                                                   \
        if (e_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(e_));                                     \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

// ------------------------------------------------------------- configuración
struct Config {
    std::string precision = "fp32";    // fp16 | fp32
    std::string medida = "kernel";     // kernel | end-to-end
    int n = 100;
    long reps = 10000;
    int corridas = 5;                  // mediciones válidas (sin contar calentamiento)
    int hilos_max = 256;               // hilos máximos por bloque
    uint32_t semilla = 42;
    std::string csv;
};

static void uso(const char* prog) {
    std::printf(
        "Uso: %s --precision fp16|fp32 --n N --reps R [opciones]\n"
        "  --measure kernel|end-to-end  qué se mide (por defecto: kernel)\n"
        "  --runs K                     mediciones tras el calentamiento (por defecto: 5)\n"
        "  --block B                    hilos máximos por bloque, múltiplo de 32 (por defecto: 256)\n"
        "  --seed S                     semilla de la entrada (por defecto: 42)\n"
        "  --csv archivo.csv            agrega una fila por medición\n", prog);
}

static Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto valor = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "Falta valor para %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if      (a == "--precision") c.precision = valor();
        else if (a == "--n")         c.n = std::atoi(valor().c_str());
        else if (a == "--reps")      c.reps = std::atol(valor().c_str());
        else if (a == "--runs")      c.corridas = std::atoi(valor().c_str());
        else if (a == "--block")     c.hilos_max = std::atoi(valor().c_str());
        else if (a == "--seed")      c.semilla = (uint32_t)std::strtoul(valor().c_str(), nullptr, 10);
        else if (a == "--measure")   c.medida = valor();
        else if (a == "--csv")       c.csv = valor();
        else if (a == "-h" || a == "--help") { uso(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "Opción desconocida: %s\n", a.c_str()); uso(argv[0]); std::exit(1); }
    }
    if ((c.precision != "fp16" && c.precision != "fp32") ||
        (c.medida != "kernel" && c.medida != "end-to-end") ||
        c.n < 1 || c.reps < 1 || c.corridas < 1 ||
        c.hilos_max < 32 || c.hilos_max > 1024 || c.hilos_max % 32 != 0) {
        uso(argv[0]);
        std::exit(1);
    }
    return c;
}

// ------------------------------------------------------- operaciones por tipo
template <typename T> struct Prec;

template <> struct Prec<float> {
    __device__ static float a_float(float v) { return v; }
    __device__ static float de_float(float v) { return v; }
    __device__ static float exp_resta(float x, float m) { return expf(x - m); }
    __device__ static float dividir(float e, float s) { return e / s; }
};

template <> struct Prec<__half> {
    __device__ static float a_float(__half v) { return __half2float(v); }
    __device__ static __half de_float(float v) { return __float2half(v); }
    __device__ static __half exp_resta(__half x, __half m) { return hexp(__hsub(x, m)); }
    __device__ static __half dividir(__half e, __half s) { return __hdiv(e, s); }
};

struct OpMax {
    __device__ static float neutro() { return -INFINITY; }
    __device__ static float f(float a, float b) { return fmaxf(a, b); }
};
struct OpSuma {
    __device__ static float neutro() { return 0.0f; }
    __device__ static float f(float a, float b) { return a + b; }
};

// Reducción de bloque; devuelve el resultado a todos los hilos.
// Requiere blockDim.x múltiplo de 32 (todos los hilos de cada warp activos).
template <typename Op>
__device__ float reducir_bloque(float v, float* compartida) {
    const int carril = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    for (int d = 16; d > 0; d >>= 1) v = Op::f(v, __shfl_down_sync(0xffffffffu, v, d));
    __syncthreads();                       // la memoria compartida pudo usarse antes
    if (carril == 0) compartida[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (carril < nwarps) ? compartida[carril] : Op::neutro();
        for (int d = 16; d > 0; d >>= 1) v = Op::f(v, __shfl_down_sync(0xffffffffu, v, d));
        if (carril == 0) compartida[0] = v;
    }
    __syncthreads();
    return compartida[0];
}

template <typename T>
__global__ void softmax_kernel(const T* __restrict__ x, T* __restrict__ y, int n) {
    typedef Prec<T> P;
    __shared__ float compartida[32];
    const int tid = threadIdx.x, paso = blockDim.x;

    // 1) máximo
    float m = -INFINITY;
    for (int i = tid; i < n; i += paso) m = fmaxf(m, P::a_float(x[i]));
    m = reducir_bloque<OpMax>(m, compartida);
    const T m_t = P::de_float(m);          // exacto: m es uno de los x_i

    // 2) exponencial y suma
    float s = 0.0f;
    for (int i = tid; i < n; i += paso) {
        T e = P::exp_resta(x[i], m_t);
        y[i] = e;
        s += P::a_float(e);
    }
    s = reducir_bloque<OpSuma>(s, compartida);
    const T s_t = P::de_float(s);

    // 3) normalización (cada hilo relee los e_i que él mismo escribió)
    for (int i = tid; i < n; i += paso) y[i] = P::dividir(y[i], s_t);
}

// ------------------------------------------------------------------ entrada
// Mismo generador que softmax_cpu.cpp: mt19937 y mapeo explícito a [-4, 4).
static void generar_entrada(std::vector<float>& x, uint32_t semilla) {
    std::mt19937 gen(semilla);
    for (size_t i = 0; i < x.size(); ++i) {
        double u = gen() / 4294967296.0;
        x[i] = (float)(-4.0 + 8.0 * u);
    }
}

static void softmax_ref(const std::vector<float>& x, std::vector<double>& y) {
    double m = x[0];
    for (float v : x) m = std::max(m, (double)v);
    double s = 0.0;
    for (size_t i = 0; i < x.size(); ++i) { y[i] = std::exp((double)x[i] - m); s += y[i]; }
    for (double& v : y) v /= s;
}

// Conversiones en el host
static inline float a_float_host(float v) { return v; }
static inline float a_float_host(__half v) { return __half2float(v); }
template <typename T> static inline T de_float_host(float v);
template <> inline float de_float_host<float>(float v) { return v; }
template <> inline __half de_float_host<__half>(float v) { return __float2half(v); }

// ---------------------------------------------------------------- utilidades
static double ahora_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
static double epoch_s() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}
static std::string leer_linea(const std::string& ruta) {
    std::ifstream f(ruta);
    std::string s;
    if (f) std::getline(f, s);
    return s;
}
static double frecuencia_gpu_mhz() {
    const char* rutas[] = {"/sys/class/devfreq/57000000.gpu/cur_freq",
                           "/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq"};
    for (const char* r : rutas) {
        std::string s = leer_linea(r);
        if (!s.empty()) return std::atof(s.c_str()) / 1e6;
    }
    return NAN;
}
static double temperatura_c(const char* clave) {
    for (int z = 0; z < 16; ++z) {
        std::string base = "/sys/devices/virtual/thermal/thermal_zone" + std::to_string(z);
        std::string tipo = leer_linea(base + "/type");
        if (!tipo.empty() && tipo.find(clave) != std::string::npos) {
            std::string t = leer_linea(base + "/temp");
            if (!t.empty()) return std::atof(t.c_str()) / 1000.0;
        }
    }
    return NAN;
}

struct Verificacion {
    double suma, err_abs_max, err_abs_medio, err_rel_max, err_rel_medio;
    bool rango_ok, suma_ok, argmax_ok, empate;
    int idx_ref, idx_obt;
};

template <typename T>
static Verificacion verificar(const std::vector<T>& y, const std::vector<double>& ref, double tol_suma) {
    Verificacion v{};
    const int n = (int)y.size();
    v.rango_ok = true;
    for (int i = 0; i < n; ++i) {
        double yi = a_float_host(y[i]);
        double ea = std::fabs(yi - ref[i]);
        double er = ea / ref[i];
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
    for (int i = 1; i < n; ++i) if (a_float_host(y[i]) > a_float_host(y[v.idx_obt])) v.idx_obt = i;
    v.empate = (v.idx_obt != v.idx_ref) && (a_float_host(y[v.idx_ref]) == a_float_host(y[v.idx_obt]));
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
    const size_t bytes = (size_t)n * sizeof(T);
    const double tol_suma = (c.precision == "fp16") ? 5e-3 : 1e-5;
    int hilos = ((n + 31) / 32) * 32;
    hilos = std::max(32, std::min(hilos, c.hilos_max));

    CK(cudaFree(0));                       // crea el contexto fuera de la medición
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int rt = 0, drv = 0;
    CK(cudaRuntimeGetVersion(&rt));
    CK(cudaDriverGetVersion(&drv));

    std::printf("=== softmax_cuda | %s | N=%d | reps=%ld | medida=%s | semilla=%u ===\n",
                c.precision.c_str(), n, c.reps, c.medida.c_str(), c.semilla);
    std::printf("GPU: %s, sm_%d%d, %d SM | CUDA runtime %d.%d, driver %d.%d | %.0f MHz | temp GPU %.1f C\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, rt / 1000, (rt % 1000) / 10,
                drv / 1000, (drv % 1000) / 10, frecuencia_gpu_mhz(), temperatura_c("GPU"));
    std::printf("Lanzamiento: 1 bloque x %d hilos | memoria compartida: %zu B | datos: %zu B por vector\n",
                hilos, 32 * sizeof(float), bytes);

    // Entrada y referencia
    std::vector<float> xf(n);
    generar_entrada(xf, c.semilla);
    std::vector<double> ref(n);
    softmax_ref(xf, ref);

    // Memoria fija (pinned) en el host para que las copias no pasen por un búfer intermedio
    T *h_x, *h_y, *d_x, *d_y;
    CK(cudaMallocHost(&h_x, bytes));
    CK(cudaMallocHost(&h_y, bytes));
    CK(cudaMalloc(&d_x, bytes));
    CK(cudaMalloc(&d_y, bytes));
    for (int i = 0; i < n; ++i) h_x[i] = de_float_host<T>(xf[i]);
    CK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));

    struct Fila { double total, por_llamada_us, t_ini, t_fin, frec, temp; };
    std::vector<Fila> filas;

    for (int r = 0; r <= c.corridas; ++r) {        // r = 0 es el calentamiento
        double t_ini = epoch_s();
        double total;
        if (c.medida == "kernel") {
            // Lanzamientos consecutivos del kernel, medidos con eventos CUDA
            CK(cudaEventRecord(ev0));
            for (long k = 0; k < c.reps; ++k) softmax_kernel<T><<<1, hilos>>>(d_x, d_y, n);
            CK(cudaEventRecord(ev1));
            CK(cudaEventSynchronize(ev1));
            CK(cudaGetLastError());
            float ms = 0.0f;
            CK(cudaEventElapsedTime(&ms, ev0, ev1));
            total = ms / 1000.0;
        } else {
            // De extremo a extremo: copia H->D, lanzamiento, copia D->H.
            // La copia D->H es síncrona: espera a que termine el kernel.
            double t0 = ahora_s();
            for (long k = 0; k < c.reps; ++k) {
                CK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
                softmax_kernel<T><<<1, hilos>>>(d_x, d_y, n);
                CK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
            }
            CK(cudaGetLastError());
            total = ahora_s() - t0;
        }
        double t_fin = epoch_s();
        Fila f{total, total / c.reps * 1e6, t_ini, t_fin, frecuencia_gpu_mhz(), temperatura_c("GPU")};
        if (r == 0) {
            std::printf("Calentamiento: %.3f ms total, %.3f us/llamada\n", total * 1e3, f.por_llamada_us);
        } else {
            std::printf("Medición %d: %.3f ms total, %.4f us/llamada | GPU %.0f MHz\n",
                        r, total * 1e3, f.por_llamada_us, f.frec);
            filas.push_back(f);
        }
    }

    // Resultado final para verificar
    CK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    std::vector<T> y(h_y, h_y + n);

    std::vector<double> t;
    for (auto& f : filas) t.push_back(f.por_llamada_us);
    std::printf("Tiempo por llamada (us): mediana %.4f | mín %.4f | máx %.4f\n",
                mediana(t), *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()));

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

    if (!c.csv.empty()) {
        bool nuevo = leer_linea(c.csv).empty();
        std::ofstream f(c.csv, std::ios::app);
        if (nuevo)
            f << "plataforma,precision,medida,n,reps,medicion,hilos,tiempo_total_s,tiempo_llamada_us,"
                 "t_inicio_epoch,t_fin_epoch,frec_gpu_mhz,temp_gpu_c,suma,err_abs_max,err_abs_medio,"
                 "err_rel_max,err_rel_medio,verificacion\n";
        char buf[512];
        for (size_t i = 0; i < filas.size(); ++i) {
            const Fila& r = filas[i];
            std::snprintf(buf, sizeof buf,
                          "gpu,%s,%s,%d,%ld,%zu,%d,%.9f,%.6f,%.3f,%.3f,%.1f,%.1f,%.9f,%.6e,%.6e,%.6e,%.6e,%s\n",
                          c.precision.c_str(), c.medida.c_str(), n, c.reps, i + 1, hilos, r.total,
                          r.por_llamada_us, r.t_ini, r.t_fin, r.frec, r.temp, v.suma, v.err_abs_max,
                          v.err_abs_medio, v.err_rel_max, v.err_rel_medio, ok ? "correcto" : "incorrecto");
            f << buf;
        }
        std::printf("CSV: %zu filas agregadas a %s\n", filas.size(), c.csv.c_str());
    }

    CK(cudaEventDestroy(ev0));
    CK(cudaEventDestroy(ev1));
    CK(cudaFree(d_x));
    CK(cudaFree(d_y));
    CK(cudaFreeHost(h_x));
    CK(cudaFreeHost(h_y));
    return ok ? 0 : 2;
}

int main(int argc, char** argv) {
    Config c = parse_args(argc, argv);
    return (c.precision == "fp16") ? ejecutar<__half>(c) : ejecutar<float>(c);
}
