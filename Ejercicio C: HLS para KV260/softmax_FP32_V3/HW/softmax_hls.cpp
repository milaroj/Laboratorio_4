#include "softmax_hls.hpp"

#include <hls_math.h>
#include <hls_stream.h>


// ETAPA 1: Reducción del máximo
static void calcular_maximo(
    const data_t x[1000],
    data_t &maximo,
    int n)
{
    #pragma HLS INLINE off

    data_t m = x[0];

    for (int i = 1; i < n; i++) {
        #pragma HLS PIPELINE II=1

        if (x[i] > m) {
            m = x[i];
        }
    }

    maximo = m;
}


// ETAPA 2: Cálculo de exponenciales
static void calcular_exponenciales(
    const data_t x[1000],
    data_t exp_valores[1000],
    hls::stream<data_t> &flujo_exp,
    data_t maximo,
    int n)
{
    #pragma HLS INLINE off

    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1

        data_t valor = hls::exp(x[i] - maximo);

        exp_valores[i] = valor;

        flujo_exp.write(valor);
    }
}


// ETAPA 3: Reducción de la suma
static void calcular_suma(
    hls::stream<data_t> &flujo_exp,
    data_t &suma,
    int n)
{
    #pragma HLS INLINE off

    data_t acumulador = 0;

    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1

        acumulador += flujo_exp.read();
    }

    suma = acumulador;
}


// ETAPA 4: Normalización
static void normalizar(
    const data_t exp_valores[1000],
    data_t y[1000],
    data_t suma,
    int n)
{
    #pragma HLS INLINE off

    for (int i = 0; i < n; i++) {
        #pragma HLS PIPELINE II=1

        y[i] = exp_valores[i] / suma;
    }
}


// FUNCIÓN SUPERIOR: SOFTMAX HLS
void softmax_hls(
    const data_t x[1000],
    data_t y[1000],
    int n)
{
    if (n <= 0 || n > 1000) {
        return;
    }

    data_t maximo;
    data_t suma;

    // Almacenamiento interno de exponenciales
    data_t exp_valores[1000];

    #pragma HLS ARRAY_PARTITION variable=exp_valores cyclic factor=4 dim=1

    // Comunicación entre exponenciales y suma
    hls::stream<data_t> flujo_exp("flujo_exp");

    #pragma HLS STREAM variable=flujo_exp depth=1024

    // Ejecución concurrente de etapas
    #pragma HLS DATAFLOW

    calcular_maximo(
        x,
        maximo,
        n
    );

    calcular_exponenciales(
        x,
        exp_valores,
        flujo_exp,
        maximo,
        n
    );

    calcular_suma(
        flujo_exp,
        suma,
        n
    );

    normalizar(
        exp_valores,
        y,
        suma,
        n
    );
}