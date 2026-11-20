#include "softmax_hls.hpp"
#include <hls_math.h>

void softmax_hls(const data_t x[1000],
                 data_t y[1000],
                 int n)
{
    if (n <= 0 || n > 1000)
        return;

    data_t maximo = x[0];
    float suma = 0.0f;

    data_t exp_valores[1000];

    // 1. Reducción del máximo
    for (int i = 1; i < n; i++) {
        #pragma HLS PIPELINE II=1

        if (x[i] > maximo) {
            maximo = x[i];
        }
    }

    // 2. Exponenciales y reducción de la suma
    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1

        const data_t diferencia = x[i] - maximo;
        exp_valores[i] = hls::exp(diferencia);
        suma += static_cast<float>(exp_valores[i]);
    }

    // 3. Normalización
    const data_t denominador = static_cast<data_t>(suma);
    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1

        y[i] = exp_valores[i] / denominador;
    }
}