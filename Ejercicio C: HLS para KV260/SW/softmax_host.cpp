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
#include <utility>
#include <vector>

#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// V0/V1 usan: x, y, n.
// V2/V3 usan: x_0, x_1, y, n.

constexpr int kMaxElements = 1000;
using Clock = std::chrono::steady_clock;
using EpochClock = std::chrono::system_clock;

static double elapsedUs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

static void generateInput(float* input, int n) {
    std::uint32_t seed = 42;
    for (int i = 0; i < n; ++i) {
        seed = 1664525u * seed + 1013904223u;
        const float u = (seed >> 8) / 16777216.0f;
        input[i] = -4.0f + 8.0f * u;
    }
}

static void computeSoftmaxReference(const float* input, double* output, int n) {
    double maximum = static_cast<double>(input[0]);
    for (int i = 1; i < n; ++i) {
        maximum = std::max(maximum, static_cast<double>(input[i]));
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        output[i] = std::exp(static_cast<double>(input[i]) - maximum);
        sum += output[i];
    }
    for (int i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

static std::string findPowerSensor() {
    namespace fs = std::filesystem;
    for (const auto& entry : fs::directory_iterator("/sys/class/hwmon")) {
        std::ifstream file(entry.path() / "name");
        std::string name;
        if (file >> name && name.find("ina260") != std::string::npos) {
            const auto path = entry.path() / "power1_input";
            if (fs::exists(path)) return path.string();
        }
    }
    throw std::runtime_error("No se encontro INA260 / power1_input");
}

static double readPowerWatts(const std::string& path) {
    std::ifstream file(path);
    double microwatts = 0.0;
    if (!(file >> microwatts)) {
        throw std::runtime_error("No se pudo leer la potencia del INA260");
    }
    return microwatts / 1'000'000.0;
}

class PowerMonitor {
    std::string path_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::vector<double> readings_;

public:
    explicit PowerMonitor(std::string path) : path_(std::move(path)) {}

    void start() {
        readings_.clear();
        readings_.push_back(readPowerWatts(path_));
        running_.store(true);
        worker_ = std::thread([this] {
            while (running_.load()) {
                try {
                    readings_.push_back(readPowerWatts(path_));
                } catch (...) {
                    // Ignore an individual failed sensor read.
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    void stop() {
        running_.store(false);
        if (worker_.joinable()) worker_.join();
    }

    ~PowerMonitor() { stop(); }

    std::size_t sampleCount() const { return readings_.size(); }

    double averageWatts() const {
        if (readings_.empty()) return 0.0;
        return std::accumulate(readings_.begin(), readings_.end(), 0.0) /
               readings_.size();
    }
};

static std::string detectVariant(const std::string& path) {
    for (int version = 0; version <= 3; ++version) {
        if (path.find("softmax_v" + std::to_string(version) + "_fp32") !=
            std::string::npos) {
            return "V" + std::to_string(version);
        }
    }
    throw std::invalid_argument("Nombre del xclbin sin variante reconocida (V0..V3)");
}

static std::string runIdentifier() {
    const auto duration = EpochClock::now().time_since_epoch();
    return std::to_string(
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
}

static double sampleStdDev(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double squaredSum = 0.0;
    for (double value : values) squaredSum += (value - mean) * (value - mean);
    return std::sqrt(squaredSum / (values.size() - 1));
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        std::cerr << "Uso: " << argv[0]
                  << " <archivo.xclbin> [repeticiones=100]"
                  << " [N=all|10|100|500|1000] [warmup=3]\n";
        return 1;
    }

    try {
        const std::string xclbinPath = argv[1];
        const std::string variant = detectVariant(xclbinPath);
        const bool partitionedInput = (variant == "V2" || variant == "V3");
        const int repetitions = argc >= 3 ? std::stoi(argv[2]) : 100;
        const std::string nSelection = argc >= 4 ? argv[3] : "all";
        const int warmup = argc >= 5 ? std::stoi(argv[4]) : 3;
        if (repetitions <= 0 || warmup < 0) {
            throw std::invalid_argument("Repeticiones > 0 y warmup >= 0");
        }

        std::vector<int> sizes = {10, 100, 500, 1000};
        if (nSelection != "all") {
            const int selected = std::stoi(nSelection);
            if (std::find(sizes.begin(), sizes.end(), selected) == sizes.end()) {
                throw std::invalid_argument("N debe ser 10, 100, 500, 1000 o all");
            }
            sizes = {selected};
        }

        const std::string base =
            "resultados_softmax_fp32_" + variant + "_" + runIdentifier();
        std::ofstream summary(base + "_resumen.csv");
        std::ofstream calls(base + "_llamadas.csv");
        if (!summary || !calls) {
            throw std::runtime_error("No se pudieron crear los CSV");
        }
        summary << std::setprecision(12);
        calls << std::setprecision(12);

        // Preserve the CSV schema used by the previous host.
        summary << "variante,precision,n,reps,warmup,h2d_us,kernel_xrt_lote_us,"
                   "kernel_xrt_us_por_llamada,d2h_us,total_lote_us,"
                   "total_us_por_llamada_amortizado,potencia_promedio_w,"
                   "energia_lote_uj,energia_por_llamada_uj,energia_por_elemento_j,"
                   "muestras_potencia,error_max,rmse,suma_salida,estado,"
                   "media_tiempo_llamada_us,desv_tiempo_llamada_us\n";
        calls << "variante,precision,n,numero_llamada,kernel_xrt_us\n";

        auto device = xrt::device(0);
        auto uuid = device.load_xclbin(xclbinPath);
        auto kernel = xrt::kernel(device, uuid, "softmax_hls");

        // In V2/V3, the HLS cyclic factor-2 partition turns x into
        // two separate M_AXI arguments (x_0 and x_1).
        const std::size_t firstCapacity = static_cast<std::size_t>(
            partitionedInput ? (kMaxElements + 1) / 2 : kMaxElements);
        xrt::bo input0(device, firstCapacity * sizeof(float), kernel.group_id(0));
        xrt::bo input1;
        if (partitionedInput) {
            input1 = xrt::bo(device,
                             (kMaxElements / 2) * sizeof(float),
                             kernel.group_id(1));
        }
        const int outputArg = partitionedInput ? 2 : 1;
        xrt::bo output(device,
                       kMaxElements * sizeof(float),
                       kernel.group_id(outputArg));

        float* mappedInput0 = input0.map<float*>();
        float* mappedInput1 = partitionedInput ? input1.map<float*>() : nullptr;
        float* mappedOutput = output.map<float*>();
        if (!mappedInput0 || !mappedOutput ||
            (partitionedInput && !mappedInput1)) {
            throw std::runtime_error("Error al mapear buffers");
        }

        const std::string sensor = findPowerSensor();
        std::cout << "Sensor: " << sensor << '\n'
                  << "Variante: " << variant
                  << " | Repeticiones: " << repetitions
                  << " | Warm-up: " << warmup << '\n'
                  << "Interfaz: "
                  << (partitionedInput ? "x_0, x_1, y, n" : "x, y, n") << '\n';

        bool allPassed = true;
        std::array<float, kMaxElements> input{};
        std::array<double, kMaxElements> reference{};

        for (int n : sizes) {
            generateInput(input.data(), n);
            computeSoftmaxReference(input.data(), reference.data(), n);

            // Pack cyclic partitions: x_0 gets even indices, x_1 odd indices.
            // Packing is deliberately excluded from the measured H2D interval.
            if (partitionedInput) {
                for (int i = 0; i < n; ++i) {
                    if ((i & 1) == 0) {
                        mappedInput0[i / 2] = input[i];
                    } else {
                        mappedInput1[i / 2] = input[i];
                    }
                }
            } else {
                std::copy_n(input.data(), n, mappedInput0);
            }

            const std::size_t input0Bytes =
                static_cast<std::size_t>(partitionedInput ? (n + 1) / 2 : n) *
                sizeof(float);
            const std::size_t input1Bytes =
                static_cast<std::size_t>(n / 2) * sizeof(float);
            const std::size_t outputBytes =
                static_cast<std::size_t>(n) * sizeof(float);

            auto syncInputs = [&] {
                input0.sync(XCL_BO_SYNC_BO_TO_DEVICE, input0Bytes, 0);
                if (partitionedInput) {
                    input1.sync(XCL_BO_SYNC_BO_TO_DEVICE, input1Bytes, 0);
                }
            };

            auto runOnce = [&] {
                if (partitionedInput) {
                    auto run = kernel(input0, input1, output, n);
                    run.wait();
                } else {
                    auto run = kernel(input0, output, n);
                    run.wait();
                }
            };

            // Warm-up is excluded from all host timers and power sampling.
            syncInputs();
            for (int j = 0; j < warmup; ++j) runOnce();

            PowerMonitor power(sensor);
            power.start();
            const auto totalStart = Clock::now();

            const auto h2dStart = Clock::now();
            syncInputs();
            const auto h2dEnd = Clock::now();

            std::vector<double> callDurations;
            callDurations.reserve(repetitions);
            const auto kernelStart = Clock::now();
            for (int j = 0; j < repetitions; ++j) {
                const auto callStart = Clock::now();
                runOnce();
                const auto callEnd = Clock::now();
                callDurations.push_back(elapsedUs(callStart, callEnd));
            }
            const auto kernelEnd = Clock::now();

            const auto d2hStart = Clock::now();
            output.sync(XCL_BO_SYNC_BO_FROM_DEVICE, outputBytes, 0);
            const auto d2hEnd = Clock::now();
            const auto totalEnd = Clock::now();
            power.stop();

            // Compare the last device output with the FP64 reference.
            double maxError = 0.0;
            double sumSquaredError = 0.0;
            double outputSum = 0.0;
            bool passed = true;
            for (int i = 0; i < n; ++i) {
                const double value = static_cast<double>(mappedOutput[i]);
                if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
                    passed = false;
                }
                const double error = std::abs(value - reference[i]);
                maxError = std::max(maxError, error);
                sumSquaredError += error * error;
                outputSum += value;
            }
            const double rmse = std::sqrt(sumSquaredError / n);
            if (maxError > 1e-5 || std::abs(outputSum - 1.0) > 1e-4) {
                passed = false;
            }
            allPassed &= passed;

            const double h2dUs = elapsedUs(h2dStart, h2dEnd);
            const double kernelBatchUs = elapsedUs(kernelStart, kernelEnd);
            const double d2hUs = elapsedUs(d2hStart, d2hEnd);
            const double totalUs = elapsedUs(totalStart, totalEnd);
            const double avgPowerWatts = power.averageWatts();
            // W * us = microjoules; this is the power seen by the INA260.
            const double batchEnergyUj = avgPowerWatts * totalUs;
            const double energyPerCallUj = batchEnergyUj / repetitions;
            const double energyPerElementJ = energyPerCallUj * 1e-6 / n;
            const double avgCallUs =
                std::accumulate(callDurations.begin(), callDurations.end(), 0.0) /
                callDurations.size();
            const double stdDevCallUs = sampleStdDev(callDurations);

            // Write CSV data after all timed measurements have completed.
            for (std::size_t j = 0; j < callDurations.size(); ++j) {
                calls << variant << ",FP32," << n << ',' << (j + 1)
                      << ',' << callDurations[j] << '\n';
            }
            summary << variant << ",FP32," << n << ',' << repetitions << ','
                    << warmup << ',' << h2dUs << ',' << kernelBatchUs << ','
                    << kernelBatchUs / repetitions << ',' << d2hUs << ','
                    << totalUs << ',' << totalUs / repetitions << ','
                    << avgPowerWatts << ',' << batchEnergyUj << ','
                    << energyPerCallUj << ',' << energyPerElementJ << ','
                    << power.sampleCount() << ',' << maxError << ',' << rmse << ','
                    << outputSum << ',' << (passed ? "PASS" : "FAIL") << ','
                    << avgCallUs << ',' << stdDevCallUs << '\n';

            std::cout << std::fixed << std::setprecision(3)
                      << "\nN=" << n
                      << " | H2D=" << h2dUs << " us"
                      << " | Kernel+XRT=" << kernelBatchUs / repetitions << " us/llamada"
                      << " | D2H=" << d2hUs << " us"
                      << " | Total=" << totalUs / repetitions << " us/llamada\n"
                      << "Potencia=" << avgPowerWatts << " W ("
                      << power.sampleCount() << " muestras)"
                      << " | Energia=" << energyPerCallUj << " uJ/llamada\n"
                      << std::scientific
                      << "Energia/elemento=" << energyPerElementJ
                      << " J | Error max=" << maxError << std::fixed
                      << " | " << (passed ? "PASS" : "FAIL") << '\n';
            if (power.sampleCount() < 10) {
                std::cout << "AVISO: pocas muestras de potencia; "
                             "interpretar energia con cautela.\n";
            }
        }

        std::cout << "\nCSV resumen: " << base << "_resumen.csv\n"
                  << "CSV llamadas: " << base << "_llamadas.csv\n"
                  << "NOTA: latencia interna IP, II, frecuencia HLS y recursos "
                     "NO son datos del host.\n";
        return allPassed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}