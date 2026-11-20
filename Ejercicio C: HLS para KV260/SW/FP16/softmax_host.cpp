#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// Host para el laboratorio Softmax FP16 en KV260.
// Metricas del host: H2D, Kernel+XRT, D2H, total, potencia, energia estimada y error.
// La latencia interna del IP se obtiene APARTE de las trazas ACCEL_MONITOR de XRT.
// Los recursos, II y frecuencia estimada se obtienen de los reportes de Vitis HLS.

constexpr int MAX_N = 1000;
using fp16_t = __fp16;
static_assert(sizeof(fp16_t) == 2);
using Reloj = std::chrono::steady_clock;
using RelojEpoch = std::chrono::system_clock;

static double micros(Reloj::time_point a, Reloj::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

static void generar_entradas(float* x, int n) {
    uint32_t semilla = 42;
    for (int i = 0; i < n; ++i) {
        semilla = 1664525u * semilla + 1013904223u;
        const float u = (semilla >> 8) / 16777216.0f;
        x[i] = -4.0f + 8.0f * u;
    }
}

static void softmax_referencia(const float* x, double* y, int n) {
    double maximo = static_cast<double>(x[0]);
    for (int i = 1; i < n; ++i)
        maximo = std::max(maximo, static_cast<double>(x[i]));

    double suma = 0.0;
    for (int i = 0; i < n; ++i) {
        y[i] = std::exp(static_cast<double>(x[i]) - maximo);
        suma += y[i];
    }
    for (int i = 0; i < n; ++i)
        y[i] /= suma;
}

static std::string buscar_sensor() {
    namespace fs = std::filesystem;
    for (const auto& entrada : fs::directory_iterator("/sys/class/hwmon")) {
        std::ifstream archivo(entrada.path() / "name");
        std::string nombre;
        if (archivo >> nombre && nombre.find("ina260") != std::string::npos) {
            const auto ruta = entrada.path() / "power1_input";
            if (fs::exists(ruta)) return ruta.string();
        }
    }
    throw std::runtime_error("No se encontro INA260 / power1_input");
}

static double leer_potencia(const std::string& ruta) {
    std::ifstream archivo(ruta);
    double microwatts = 0.0;
    if (!(archivo >> microwatts))
        throw std::runtime_error("No se pudo leer la potencia del INA260");
    return microwatts / 1'000'000.0;
}

// El sensor se muestrea de forma independiente a la ejecucion del kernel.
// Para lotes muy cortos puede generar pocas muestras: revisar muestras_potencia.
class MonitorPotencia {
    std::string ruta_;
    std::atomic<bool> activo_{false};
    std::thread trabajador_;
    std::vector<double> muestras_;
public:
    explicit MonitorPotencia(std::string ruta) : ruta_(std::move(ruta)) {}
    void iniciar() {
        muestras_.clear();
        muestras_.push_back(leer_potencia(ruta_));
        activo_.store(true);
        trabajador_ = std::thread([this] {
            while (activo_.load()) {
                try { muestras_.push_back(leer_potencia(ruta_)); }
                catch (...) { /* Una lectura fallida no invalida las anteriores. */ }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    void detener() {
        activo_.store(false);
        if (trabajador_.joinable()) trabajador_.join();
    }
    ~MonitorPotencia() { detener(); }
    size_t cantidad() const { return muestras_.size(); }
    double promedio() const {
        if (muestras_.empty()) return 0.0;
        return std::accumulate(muestras_.begin(), muestras_.end(), 0.0)
               / muestras_.size();
    }
};

static std::string variante_desde_archivo(const std::string& ruta) {
    for (int v = 0; v <= 3; ++v) {
        if (ruta.find("softmax_v" + std::to_string(v) + "_fp16") != std::string::npos)
            return "V" + std::to_string(v);
    }
    throw std::invalid_argument("Nombre del xclbin sin variante reconocida (V0..V3)");
}

static std::string identificador_ejecucion() {
    const auto ahora = RelojEpoch::now().time_since_epoch();
    return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(ahora).count());
}

static double desviacion_muestral(const std::vector<double>& datos) {
    if (datos.size() < 2) return 0.0;
    const double media = std::accumulate(datos.begin(), datos.end(), 0.0) / datos.size();
    double suma = 0.0;
    for (double dato : datos) suma += (dato - media) * (dato - media);
    return std::sqrt(suma / (datos.size() - 1));
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Uso: " << argv[0]
                  << " <archivo.xclbin> [repeticiones=100] [N=all|10|100|500|1000] [warmup=3]\n";
        return 1;
    }

    try {
        const std::string xclbin = argv[1];
        const std::string variante = variante_desde_archivo(xclbin);
        const int repeticiones = argc >= 3 ? std::stoi(argv[2]) : 100;
        const std::string eleccion_n = argc >= 4 ? argv[3] : "all";
        const int calentamiento = argc >= 5 ? std::stoi(argv[4]) : 3;
        if (repeticiones <= 0 || calentamiento < 0)
            throw std::invalid_argument("Repeticiones > 0 y warmup >= 0");

        std::vector<int> tamanos = {10, 100, 500, 1000};
        if (eleccion_n != "all") {
            const int n = std::stoi(eleccion_n);
            if (std::find(tamanos.begin(), tamanos.end(), n) == tamanos.end())
                throw std::invalid_argument("N debe ser 10, 100, 500, 1000 o all");
            tamanos = {n};
        }

        // Cada ejecucion guarda los resultados con un identificador propio.
        // Para asociar summary.csv / device_trace_0.csv, ejecutar UN N por proceso.
        const std::string id = identificador_ejecucion();
        const std::string base = "resultados_softmax_fp16_" + variante + "_" + id;
        std::ofstream resumen(base + "_resumen.csv");
        std::ofstream llamadas(base + "_llamadas.csv");
        if (!resumen || !llamadas)
            throw std::runtime_error("No se pudieron crear los CSV");
        resumen << std::setprecision(12);
        llamadas << std::setprecision(12);

        resumen << "variante,precision,n,reps,warmup,h2d_us,kernel_xrt_lote_us,"
                   "kernel_xrt_us_por_llamada,d2h_us,total_lote_us,"
                   "total_us_por_llamada_amortizado,potencia_promedio_w,"
                   "energia_lote_uj,energia_por_llamada_uj,energia_por_elemento_j,"
                   "muestras_potencia,error_max,rmse,suma_salida,estado,"
                   "media_tiempo_llamada_us,desv_tiempo_llamada_us\n";
        llamadas << "variante,precision,n,numero_llamada,kernel_xrt_us\n";

        auto device = xrt::device(0);
        auto uuid = device.load_xclbin(xclbin);
        auto softmax = xrt::kernel(device, uuid, "softmax_hls");
        auto bo_x = xrt::bo(device, MAX_N * sizeof(fp16_t), softmax.group_id(0));
        auto bo_y = xrt::bo(device, MAX_N * sizeof(fp16_t), softmax.group_id(1));
        fp16_t* x = bo_x.map<fp16_t*>();
        fp16_t* y = bo_y.map<fp16_t*>();
        if (!x || !y) throw std::runtime_error("Error al mapear buffers");

        const std::string sensor = buscar_sensor();
        std::cout << "Sensor: " << sensor << "\n"
                  << "Variante: " << variante << " | Repeticiones: " << repeticiones
                  << " | Warm-up: " << calentamiento << "\n";

        bool todas_ok = true;
        for (int n : tamanos) {
            std::array<float, MAX_N> x_original{};
            generar_entradas(x_original.data(), n);
            for (int i = 0; i < n; ++i)
                x[i] = static_cast<fp16_t>(x_original[i]);
            std::array<double, MAX_N> referencia{};
            softmax_referencia(x_original.data(), referencia.data(), n);
            const size_t bytes = static_cast<size_t>(n) * sizeof(fp16_t);

            // Calentamiento FUERA de los cronometros del host y del monitor de potencia.
            bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
            for (int j = 0; j < calentamiento; ++j) {
                auto run = softmax(bo_x, bo_y, n);
                run.wait();
            }

            MonitorPotencia monitor(sensor);
            monitor.iniciar();
            const auto total_inicio = Reloj::now();

            const auto h0 = Reloj::now();
            bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
            const auto h1 = Reloj::now();

            std::vector<double> duraciones;
            duraciones.reserve(repeticiones);
            const auto k0 = Reloj::now();
            for (int j = 0; j < repeticiones; ++j) {
                const auto inicio_llamada = Reloj::now();
                auto run = softmax(bo_x, bo_y, n);
                run.wait();
                const auto fin_llamada = Reloj::now();
                const double tiempo = micros(inicio_llamada, fin_llamada);
                duraciones.push_back(tiempo);
            }
            const auto k1 = Reloj::now();

            const auto d0 = Reloj::now();
            bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, 0);
            const auto d1 = Reloj::now();
            const auto total_fin = Reloj::now();
            monitor.detener();

            // Validacion numerica: referencia FP64; salida del dispositivo FP16.
            double error_max = 0.0, cuadrados = 0.0, suma_y = 0.0;
            constexpr double minimo_normal_fp16 = 1.0 / 16384.0;
            double masa_subnormal_ref = 0.0;
            bool ok = true;
            for (int i = 0; i < n; ++i) {
                const double salida = static_cast<double>(y[i]);
                if (!std::isfinite(salida) || salida < 0.0 || salida > 1.0) ok = false;
                if (referencia[i] > 0.0 && referencia[i] < minimo_normal_fp16)
                    masa_subnormal_ref += referencia[i];
                const double error = std::abs(salida - referencia[i]);
                error_max = std::max(error_max, error);
                cuadrados += error * error;
                suma_y += salida;
            }
            const double rmse = std::sqrt(cuadrados / n);
            // Mismo criterio de validacion que en el testbench FP16.
            const double tolerancia_suma = masa_subnormal_ref + 1e-3;
            if (error_max > 1e-3 || std::abs(suma_y - 1.0) > tolerancia_suma)
                ok = false;
            todas_ok &= ok;

            const double h2d = micros(h0, h1);
            const double kernel_lote = micros(k0, k1);
            const double d2h = micros(d0, d1);
            const double total = micros(total_inicio, total_fin);
            const double potencia = monitor.promedio();
            // Estimacion: W * us = uJ; incluye potencia total observada por INA260.
            const double energia_lote_uj = potencia * total;
            const double energia_por_llamada_uj = energia_lote_uj / repeticiones;
            const double energia_por_elemento_j = energia_por_llamada_uj * 1e-6 / n;
            const double media_llamadas = std::accumulate(duraciones.begin(), duraciones.end(), 0.0)
                                          / duraciones.size();
            const double desv_llamadas = desviacion_muestral(duraciones);
            // Escribir las duraciones solo DESPUES de detener las mediciones.
            for (size_t j = 0; j < duraciones.size(); ++j)
                llamadas << variante << ",FP16," << n << ',' << (j + 1)
                         << ',' << duraciones[j] << '\n';

            resumen << variante << ",FP16," << n << ',' << repeticiones << ','
                    << calentamiento << ',' << h2d << ',' << kernel_lote << ','
                    << kernel_lote / repeticiones << ',' << d2h << ',' << total << ','
                    << total / repeticiones << ',' << potencia << ',' << energia_lote_uj
                    << ',' << energia_por_llamada_uj << ',' << energia_por_elemento_j
                    << ',' << monitor.cantidad() << ',' << error_max << ',' << rmse
                    << ',' << suma_y << ',' << (ok ? "PASS" : "FAIL") << ','
                    << media_llamadas << ',' << desv_llamadas << '\n';

            std::cout << std::fixed << std::setprecision(3)
                      << "\nN=" << n
                      << " | H2D=" << h2d << " us"
                      << " | Kernel+XRT=" << kernel_lote / repeticiones << " us/llamada"
                      << " | D2H=" << d2h << " us"
                      << " | Total=" << total / repeticiones << " us/llamada\n"
                      << "Potencia=" << potencia << " W (" << monitor.cantidad() << " muestras)"
                      << " | Energia=" << energia_por_llamada_uj << " uJ/llamada\n"
                      << std::scientific << "Energia/elemento=" << energia_por_elemento_j
                      << " J | Error max=" << error_max << std::fixed
                      << " | " << (ok ? "PASS" : "FAIL") << '\n';
            if (monitor.cantidad() < 10)
                std::cout << "AVISO: pocas muestras de potencia; interpretar energia con cautela.\n";
        }

        std::cout << "\nCSV resumen: " << base << "_resumen.csv\n"
                  << "CSV llamadas: " << base << "_llamadas.csv\n"
                  << "NOTA: latencia interna IP, II, frecuencia HLS y recursos NO son datos del host.\n";
        return todas_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
