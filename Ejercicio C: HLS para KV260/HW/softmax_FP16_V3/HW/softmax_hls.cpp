#include "softmax_hls.hpp"
#include <hls_math.h>
#include <hls_stream.h>

// 1. Reducción del máximo y copia local de la entrada
static void calcular_maximo(const data_t x[1000],
                            data_t x_local[1000],
                            data_t &maximo,
                            int n)
{
    #pragma HLS INLINE off

    data_t maximo_local = x[0];
    x_local[0] = maximo_local;

    for (int i = 1; i < n; i++) {
        #pragma HLS PIPELINE II=1

        const data_t valor = x[i];
        x_local[i] = valor;

        if (valor > maximo_local) {
            maximo_local = valor;
        }
    }

    maximo = maximo_local;
}

// 2. Cálculo de exponenciales
static void calcular_exponenciales(const data_t x_local[1000],
                                   data_t exp_valores[1000],
                                   hls::stream<data_t> &flujo_exp,
                                   data_t maximo,
                                   int n)
{
    #pragma HLS INLINE off

    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1

        const data_t diferencia = x_local[i] - maximo;
        const data_t valor = hls::exp(diferencia);
        exp_valores[i] = valor;
        flujo_exp.write(valor);
    }
}

// 3. Reducción de la suma
static void calcular_suma(hls::stream<data_t> &flujo_exp,
                          data_t &denominador,
                          int n)
{
    #pragma HLS INLINE off

    float suma = 0.0f;

    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1

        suma += static_cast<float>(flujo_exp.read());
    }

    denominador = static_cast<data_t>(suma);
}

// 4. Normalización
static void normalizar(const data_t exp_valores[1000],
                       data_t y[1000],
                       data_t denominador,
                       int n)
{
    #pragma HLS INLINE off

    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1

        y[i] = exp_valores[i] / denominador;
    }
}

static void softmax_dataflow(const data_t x[1000],
                             data_t y[1000],
                             int n)
{
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    // Copia local de la entrada, repartida entre dos bancos.
    data_t x_local[1000];
    #pragma HLS ARRAY_PARTITION variable=x_local cyclic factor=2 dim=1

    data_t exp_valores[1000];
    data_t maximo;
    data_t denominador;

    // Comunicación entre el cálculo de exponenciales y la suma.
    hls::stream<data_t> flujo_exp("flujo_exp");
    #pragma HLS STREAM variable=flujo_exp depth=1024

    calcular_maximo(x, x_local, maximo, n);
    calcular_exponenciales(x_local, exp_valores, flujo_exp, maximo, n);
    calcular_suma(flujo_exp, denominador, n);
    normalizar(exp_valores, y, denominador, n);
}

void softmax_hls(const data_t x[1000],
                 data_t y[1000],
                 int n)
{
    if (n <= 0 || n > 1000) {
        return;
    }

    softmax_dataflow(x, y, n);
}