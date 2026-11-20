# Compilacion del host Softmax FP16 V0 a V3 en Kria KV260 con XRT

CXX = g++

TARGET := softmax_host
SOURCE := softmax_host.cpp

CXXFLAGS := -Wall -Wextra -std=c++17 -O2 -I/usr/include/xrt -I.

LDFLAGS := -L/opt/xilinx/xrt/lib -pthread
LDLIBS := -lxrt_core -lxrt_coreutil

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) $(TARGET)
