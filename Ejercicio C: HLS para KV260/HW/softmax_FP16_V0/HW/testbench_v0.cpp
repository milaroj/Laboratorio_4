#include "softmax_hls.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>

// Genera entradas reproducibles en [-4, 4)
void generar_entradas(float x[1000], int n)
{
    uint32_t semilla = 42;

    for (int i = 0; i < n; i++) {
        semilla = 1664525u * semilla + 1013904223u;

        float u = (semilla >> 8) / 16777216.0f;

        x[i] = -4.0f + 8.0f * u;
    }
}

// Referencia Softmax en FP64
void softmax_referencia(const float x[1000],
                        double y_ref[1000],
                        int n)
{
    double maximo = static_cast<double>(x[0]);

    // Reducción del máximo
    for (int i = 1; i < n; i++) {
        if (static_cast<double>(x[i]) > maximo) {
            maximo = static_cast<double>(x[i]);
        }
    }

    double suma = 0.0;

    // Exponenciales
    for (int i = 0; i < n; i++) {
        y_ref[i] = std::exp(
            static_cast<double>(x[i]) - maximo
        );

        suma += y_ref[i];
    }

    // Normalización
    for (int i = 0; i < n; i++) {
        y_ref[i] /= suma;
    }
}

// Prueba para un tamaño N
bool probar_softmax(int n)
{
    float x_original[1000] = {};
    data_t x[1000] = {};
    data_t y[1000] = {};

    double y_ref[1000] = {};

    generar_entradas(x_original, n);
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<data_t>(x_original[i]);
    }

    // Ejecutar acelerador HLS
    softmax_hls(x, y, n);

    // Ejecutar referencia FP64
    softmax_referencia(x_original, y_ref, n);

    double error_max = 0.0;
    double error_cuadratico = 0.0;
    double suma_y = 0.0;
    int salidas_cero = 0;
    double masa_en_ceros = 0.0;
    constexpr double minimo_normal_fp16 = 1.0 / 16384.0;
    double masa_subnormal_ref = 0.0;

    bool valido = true;

    for (int i = 0; i < n; i++) {
        double resultado = static_cast<double>(static_cast<float>(y[i]));
        double error = std::fabs(resultado - y_ref[i]);

        if (y_ref[i] > 0.0 && y_ref[i] < minimo_normal_fp16) {
            masa_subnormal_ref += y_ref[i];
        }

        if (resultado == 0.0) {
            ++salidas_cero;
            masa_en_ceros += y_ref[i];
        }

        if (!std::isfinite(resultado) ||
            resultado < 0.0 ||
            resultado > 1.0) {
            valido = false;
        }

        if (error > error_max) {
            error_max = error;
        }

        error_cuadratico += error * error;
        suma_y += resultado;
    }

    double rmse = std::sqrt(error_cuadratico / n);
    double error_suma = std::fabs(suma_y - 1.0);

    // Tolerancias iniciales para FP16
    // Se admite la masa subnormal de referencia más un margen de redondeo.
    const double tolerancia_suma = masa_subnormal_ref + 1e-3;
    if (error_max > 1e-3 || error_suma > tolerancia_suma) {
        valido = false;
    }

    std::printf(
        "N=%4d | Error max=%.3e | RMSE=%.3e | "
        "Suma=%.8f | %s\n",
        n,
        error_max,
        rmse,
        suma_y,
        valido ? "PASS" : "FAIL"
    );

    std::printf(
        "         Ceros=%d | Masa FP64 en ceros=%.6e\n",
        salidas_cero,
        masa_en_ceros
    );

    return valido;
}

int main()
{
    const int tamanos[] = {10, 100, 500, 1000};

    bool resultado = true;

    std::printf("\n-----TEST SOFTMAX HLS FP16 -----\n\n");

    for (int n : tamanos) {
        if (!probar_softmax(n)) {
            resultado = false;
        }
    }

    std::printf("\nResultado final: %s\n",
                resultado ? "PASS" : "FAIL");

    return resultado ? 0 : 1;
}