# Ejercicio C: HLS para Kria KV260

## Descripción

Se conservan cuatro versiones (`V0`, `V1`, `V2`, `V3`), junto con sus códigos fuente, testbenches, configuraciones y reportes obtenidos durante el desarrollo.

El procedimiento comprende:

1. Simulación funcional en C/C++.
2. Síntesis HLS.
3. Cosimulación C/RTL.
4. Generación de los archivos del acelerador.
5. Transferencia de archivos a la Kria KV260 mediante SSH.
6. Carga del acelerador en la FPGA.
7. Ejecución mediante un programa host

---

## Organización

```text
Ejercicio C: HLS para KV260/
├── README.md
├── platform/
│   ├── kv260_custom.xpfm
│   ├── hw/
│   ├── sw/
│   ├── shell.json
│   └── pl.dtbo
├── softmax_FP32_V0/
├── softmax_FP32_V1/
├── softmax_FP32_V2/
└── softmax_FP32_V3/
```

Cada versión contiene:

```text
softmax_FP32_VN/
├── HW/
│   ├── softmax_hls.cpp
│   ├── softmax_hls.hpp
│   ├── testbench_v0.cpp
│   ├── Makefile
│   └── softmax_hls/
│       └── softmax_config.cfg
└── reportes/
    ├── sintesis/
    └── cosimulacion/
```

Donde `VN` representa cualquiera de las cuatro versiones disponibles.

---

# Parte I. Desarrollo en la laptop

## Requisitos

Para ejecutar las herramientas de desarrollo se necesita:

- Linux con **AMD Vitis/Vivado 2024.2** instalado.
- Herramientas `vitis-run`, `v++`, `make`, `emconfigutil` y `bootgen`.
- Plataforma personalizada incluida en `platform/`.
- Dependencias compatibles con la plataforma KV260.
- Espacio libre suficiente para síntesis, cosimulación e implementación.

Los procesos de generación de hardware pueden ocupar varios GB.

### Comprobación de herramientas

Desde la terminal:

```bash
command -v vitis-run
command -v v++
command -v make
command -v emconfigutil
command -v bootgen
```

Verificar la versión de Vitis:

```bash
v++ --version
```

**Importante:** los comandos siguientes deben ejecutarse desde la carpeta `HW` de la versión seleccionada.

Los archivos `.cfg` deben referenciar correctamente las fuentes y el testbench.

---

## Seleccionar la versión FP32

Desde el directorio principal del ejercicio, ingresar a la versión deseada.

Por ejemplo:

```bash
cd softmax_FP32_V0/HW
```

Para las demás versiones:

```bash
cd softmax_FP32_V1/HW
```

```bash
cd softmax_FP32_V2/HW
```

```bash
cd softmax_FP32_V3/HW
```

Los procedimientos siguientes son equivalentes para las cuatro versiones.

---

## 1. Simulación C (CSIM)

La simulación C permite comprobar el comportamiento funcional del código C/C++ mediante el testbench `testbench_v0.cpp`.

Esta etapa no genera hardware.

### Comando

```bash
vitis-run --mode hls --csim \
    --config ./softmax_hls/softmax_config.cfg \
    --work_dir ./softmax_hls
```

### Objetivo

Verificar que el acelerador Softmax produce resultados correctos antes de realizar la síntesis.

---

## 2. Síntesis HLS (CSYNTH)

La síntesis HLS transforma la descripción C/C++ del acelerador en una implementación RTL.

Durante esta etapa se generan estimaciones de:

- Latencia del acelerador.
- Intervalo de iniciación.
- Utilización de recursos FPGA.
- Frecuencia de operación estimada.

### Comando

```bash
v++ -c --mode hls \
    --config ./softmax_hls/softmax_config.cfg \
    --work_dir ./softmax_hls
```

### Objetivo

Obtener una implementación RTL del acelerador y evaluar sus características de hardware.

---

## 3. Cosimulación C/RTL (COSIM)

La cosimulación permite comparar el comportamiento de la implementación RTL sintetizada con el comportamiento esperado por el testbench C/C++.

### Comando

```bash
vitis-run --mode hls --cosim \
    --config ./softmax_hls/softmax_config.cfg \
    --work_dir ./softmax_hls
```

### Objetivo

Comprobar que la implementación de hardware conserva el comportamiento funcional esperado.

**Orden recomendado:**

```text
Simulación C
     ↓
Síntesis HLS
     ↓
Cosimulación C/RTL
```

La cosimulación requiere que el directorio de trabajo contenga el componente HLS previamente sintetizado.

---

## 4. Generación del acelerador mediante Makefile

Cada versión contiene un Makefile para construir el acelerador utilizando las herramientas de AMD Vitis.

El proceso comprende las etapas de compilación, enlace y empaquetado definidas en dicho archivo.

### Ejecutar la compilación

Desde la carpeta `HW`:

```bash
make
```

Si el Makefile define explícitamente el objetivo `build`, también puede ejecutarse:

```bash
make build
```

### Nombres de los aceleradores

| Versión | Nombre base del acelerador |
|---|---|
| `softmax_FP32_V0` | `softmax_v0_fp32` |
| `softmax_FP32_V1` | `softmax_v1_fp32` |
| `softmax_FP32_V2` | `softmax_v2_fp32` |
| `softmax_FP32_V3` | `softmax_v3_fp32` |

### Directorios de generación

Los directorios previstos son:

```text
HW/
├── tmp.hw/
├── build.hw/
└── package.hw/
```

Donde:

- `tmp.hw/`: archivos temporales de compilación.
- `build.hw/`: archivos intermedios de construcción.
- `package.hw/`: archivos finales empaquetados para el acelerador.

### Archivos del acelerador

Dependiendo del proceso de compilación y empaquetado, pueden obtenerse:

| Archivo | Descripción |
|---|---|
| `.xclbin` | Binario utilizado por XRT para identificar y gestionar el acelerador. |
| `.bit.bin` | Archivo de configuración de la lógica programable de la FPGA. |
| `.dtbo` | Device Tree Overlay que describe los componentes de hardware al sistema operativo. |
| `shell.json` | Metadatos utilizados para gestionar la aplicación FPGA en Kria. |

La presencia de cada archivo debe verificarse después de ejecutar el Makefile.

### Verificar los archivos generados

```bash
ls -lh package.hw/
```

### Inspeccionar el Makefile

Para revisar los comandos sin ejecutarlos:

```bash
make -n
```

Si existe el objetivo `build`:

```bash
make -n build
```

### Limpieza

Para eliminar los archivos definidos por el objetivo de limpieza:

```bash
make clean
```

Si el Makefile incluye `cleanall`, puede utilizarse para una limpieza más amplia:

```bash
make cleanall
```

**Precaución:** verificar los objetivos del Makefile antes de ejecutar comandos de limpieza, ya que podrían eliminar resultados cuya generación requiere mucho tiempo.

---

# Parte II. Transferencia del acelerador a Kria KV260

## 5. Requisitos en la Kria KV260

La tarjeta debe disponer de:

- Sistema operativo Linux compatible.
- Xilinx Runtime (XRT).
- Conexión de red con la laptop.
- Servidor SSH habilitado.
- Usuario con permisos para acceder al sistema.
- Herramientas necesarias para cargar la aplicación FPGA.
- Espacio disponible para almacenar los archivos del acelerador.

### Verificar XRT

Desde una terminal de la Kria:

```bash
xbutil --version
```

### Verificar el gestor de aplicaciones FPGA

```bash
command -v xmutil
```

La disponibilidad de `xmutil` depende de la imagen Linux instalada y de sus paquetes.

---

## 6. Establecer conexión SSH

La conexión SSH permite acceder remotamente a la terminal de la Kria desde la laptop.

Ambos dispositivos deben tener conectividad de red.

### Obtener la dirección IP de la Kria

Desde la terminal de la Kria:

```bash
hostname -I
```

También puede utilizarse:

```bash
ip -4 addr
```

Identificar la dirección IP correspondiente a la interfaz de red utilizada.

### Conectarse desde la laptop

```bash
ssh usuario@IP_KRIA
```

Sustituir:

- `usuario`: nombre del usuario configurado en la Kria.
- `IP_KRIA`: dirección IP de la tarjeta.

Ejemplo:

```bash
ssh ubuntu@192.168.0.100
```

La dirección del ejemplo debe reemplazarse por la dirección real de la Kria.

Si la conexión es correcta, se tendrá acceso a la terminal remota.

### Salir de la conexión SSH

```bash
exit
```

---

## 7. Transferencia de archivos mediante SCP

SCP (*Secure Copy Protocol*) permite copiar archivos entre la laptop y la Kria utilizando SSH.

Los comandos de transferencia se ejecutan desde la laptop.

### Transferir un archivo individual

Por ejemplo, para transferir el archivo `.xclbin`:

```bash
scp archivo.xclbin usuario@IP_KRIA:/home/usuario/
```

### Transferir el directorio completo de resultados

Desde la carpeta `HW` de la versión compilada:

```bash
scp -r package.hw usuario@IP_KRIA:/home/usuario/
```

La opción `-r` permite copiar recursivamente todos los archivos y subdirectorios.

### Ejemplo de transferencia

```bash
scp -r package.hw ubuntu@192.168.0.100:/home/ubuntu/
```

Este comando transfiere los resultados de compilación a la carpeta personal del usuario remoto.

### Verificar la transferencia

Ingresar a la Kria:

```bash
ssh usuario@IP_KRIA
```

Comprobar que los archivos están disponibles:

```bash
ls -lh ~/package.hw/
```

**Importante:** transferir los archivos mediante SSH no significa que la FPGA ya esté configurada.

---

# Parte III. Carga del acelerador en la FPGA

## 8. Preparar los archivos de configuración

Para utilizar el acelerador, la Kria debe disponer de los archivos correspondientes a la implementación de hardware.

En un flujo de aplicaciones FPGA de Kria gestionado mediante `xmutil`, normalmente se utilizan:

```text
nombre_acelerador/
├── nombre_acelerador.bit.bin
├── nombre_acelerador.dtbo
└── shell.json
```

Los nombres exactos y el contenido requerido dependen del paquete generado.

El archivo `.xclbin` se utiliza posteriormente desde XRT, de acuerdo con la plataforma y el procedimiento de ejecución.

---

## 9. Instalar la aplicación FPGA

En las imágenes de Kria que utilizan el gestor de aplicaciones FPGA, los archivos de configuración se almacenan habitualmente en:

```text
/lib/firmware/xilinx/
```

Cada aplicación dispone de su propio directorio.

Por ejemplo:

```text
/lib/firmware/xilinx/softmax_v0_fp32/
```

### Crear el directorio

Desde la Kria:

```bash
sudo mkdir -p /lib/firmware/xilinx/softmax_v0_fp32
```

### Copiar los archivos de configuración

Suponiendo que los archivos correspondientes se encuentran en `~/package.hw/`:

```bash
sudo cp ~/package.hw/*.bit.bin \
    /lib/firmware/xilinx/softmax_v0_fp32/
```

```bash
sudo cp ~/package.hw/*.dtbo \
    /lib/firmware/xilinx/softmax_v0_fp32/
```

```bash
sudo cp ~/package.hw/shell.json \
    /lib/firmware/xilinx/softmax_v0_fp32/
```

Estos comandos presuponen que el paquete contiene dichos archivos.

### Verificar el directorio

```bash
ls -lh /lib/firmware/xilinx/softmax_v0_fp32/
```

---

## 10. Cargar el acelerador mediante xmutil

En sistemas compatibles con el gestor de aplicaciones FPGA de Kria, `xmutil` permite consultar y cargar aplicaciones de hardware.

### Consultar aplicaciones disponibles

```bash
sudo xmutil listapps
```

La aplicación debe aparecer entre las disponibles si su paquete es reconocido correctamente.

### Cargar la aplicación

```bash
sudo xmutil loadapp softmax_v0_fp32
```

El nombre utilizado debe coincidir con el directorio de la aplicación.

### Consultar la aplicación cargada

```bash
sudo xmutil listapps
```

### Descargar la aplicación

Cuando sea necesario:

```bash
sudo xmutil unloadapp
```

**Importante:** estos comandos corresponden al flujo de carga mediante `xmutil`. Su funcionamiento requiere que la imagen Linux y los archivos de configuración sean compatibles.

---

# Parte IV. Ejecución del acelerador

## 11. Utilización del acelerador mediante XRT

Una vez cargada correctamente la configuración de hardware, el acelerador puede utilizarse mediante Xilinx Runtime (XRT).

XRT proporciona una interfaz para controlar los aceleradores FPGA desde un programa ejecutado en el procesador de la Kria.

### Archivos necesarios

Para ejecutar el acelerador normalmente se necesita:

1. El archivo `.xclbin` compatible con la configuración cargada.
2. Un programa anfitrión (*host*) compilado para la arquitectura de la Kria.
3. Las bibliotecas de XRT correspondientes.
4. Los datos de entrada para la operación Softmax.

### Funciones del programa anfitrión

El programa debe encargarse de:

- Inicializar el dispositivo FPGA mediante XRT.
- Cargar o abrir el archivo `.xclbin` correspondiente.
- Obtener acceso al kernel Softmax.
- Reservar los buffers de memoria.
- Transferir los datos de entrada.
- Iniciar la ejecución del acelerador.
- Esperar a que finalice la operación.
- Recuperar los resultados.

### Consideración importante

Los archivos de hardware no ejecutan automáticamente la operación Softmax.
Para procesar datos reales es necesario disponer de un programa anfitrión compatible con la interfaz del kernel.
El procedimiento concreto de ejecución depende de cómo se haya implementado dicho programa.

---

# Parte V. Reportes y reproducibilidad

## 12. Reportes disponibles

Los reportes obtenidos durante el desarrollo se conservan en la carpeta `reportes/` de cada versión.

### Reportes de síntesis

```text
reportes/sintesis/
```

Contienen información relacionada con:

- Latencia estimada.
- Intervalo de iniciación.
- Frecuencia de operación.
- Utilización de recursos FPGA.

### Reportes de cosimulación

```text
reportes/cosimulacion/
```

Contienen los resultados de la verificación C/RTL y las métricas obtenidas durante la cosimulación.

Los reportes corresponden a ejecuciones previas y no sustituyen una nueva validación en otro entorno.

---

## 13. Observaciones de reproducibilidad

- Los proyectos fueron desarrollados utilizando AMD Vitis/Vivado 2024.2.
- El Makefile utiliza una plataforma personalizada para Kria KV260.
- La estructura de directorios debe conservarse para evitar problemas con las rutas relativas.
- La plataforma se encuentra en `platform/kv260_custom.xpfm`.
- Los archivos `softmax_config.cfg` deben referenciar correctamente `softmax_hls.cpp`, `softmax_hls.hpp` y `testbench_v0.cpp`.
- Si una simulación presenta errores relacionados con archivos inexistentes, deben revisarse las rutas de las fuentes.
- Los resultados de síntesis y cosimulación pueden variar según la configuración y versión de las herramientas.
- La generación del acelerador puede requerir varios GB de almacenamiento.
- La transferencia mediante SSH requiere conectividad entre la laptop y la Kria.
- Los archivos `.bit.bin`, `.dtbo` y `shell.json` deben corresponder a la misma configuración de hardware.
- El archivo `.xclbin` debe ser compatible con la configuración de FPGA utilizada.
- La ejecución requiere un programa anfitrión compatible con XRT.
- No se ha verificado una reconstrucción completa del proyecto después de clonar este repositorio.

---

## 14. Resumen del flujo de trabajo

### En la laptop

```text
Código C/C++
     |
     v
Simulación C
     |
     v
Síntesis HLS
     |
     v
Cosimulación C/RTL
     |
     v
Makefile
     |
     v
Generación y empaquetado
     |
     v
Archivos del acelerador
     |
     v
Transferencia mediante SCP/SSH
```

### En la Kria KV260

```text
Recepción de archivos
     |
     v
Instalación de archivos de configuración
     |
     v
Carga de la aplicación FPGA
     |
     v
Inicialización mediante XRT
     |
     v
Ejecución del kernel Softmax
     |
     v
Recuperación de resultados
```

### Herramientas utilizadas

| Etapa | Herramienta |
|---|---|
| Simulación C | `vitis-run --mode hls --csim` |
| Síntesis HLS | `v++ -c --mode hls` |
| Cosimulación C/RTL | `vitis-run --mode hls --cosim` |
| Construcción del acelerador | `make` |
| Conexión remota | `ssh` |
| Transferencia de archivos | `scp` |
| Gestión de aplicaciones FPGA | `xmutil` |
| Ejecución del acelerador | XRT |

---
