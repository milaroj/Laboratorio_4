#include "softmax_hls.hpp"
#include <hls_math.h>


void softmax_hls(const data_t x[1000],
                 data_t y[1000],
                 int n)
{
    if (n <= 0 || n > 1000)
        return;

    data_t maximo = x[0];
    data_t suma = 0;

    data_t exp_valores[1000];

    // 1. Reducción del máximo
    for (int i = 1; i < n; i++) {
        if (x[i] > maximo) {
            maximo = x[i];
        }
    }

    // 2. Exponenciales y reducción de la suma
    for (int i = 0; i < n; i++) {
        exp_valores[i] = hls::exp(x[i] - maximo);
        suma += exp_valores[i];
    }

    // 3. Normalización
    for (int i = 0; i < n; i++) {
        y[i] = exp_valores[i] / suma;
    }
}