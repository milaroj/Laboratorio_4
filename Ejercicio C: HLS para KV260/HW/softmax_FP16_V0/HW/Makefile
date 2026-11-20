APP := softmax_v0_fp16

# Platform and input files
TARGET := hw
PLATFORM_DIR := ../../../platform
PLATFORM := platform/kv260_custom.xpfm
HLS_FILES := softmax_hls.cpp softmax_hls.hpp testbench_v0.cpp
HLS_CONFIG := softmax_hls/softmax_config.cfg

# Build directories and generated files
BUILD_DIR := $(HOME)/softmax-fp16-build/$(APP)
TEMP_DIR := $(BUILD_DIR)/tmp.$(TARGET)
SALIDA ?= $(HOME)/softmax-fp16-salida/$(APP)

HLS_XO := $(BUILD_DIR)/hls_build/softmax_hls.xo
LINK_OUTPUT := $(BUILD_DIR)/$(APP).xclbin
SYSTEM_BIT := $(BUILD_DIR)/system.bit
BOOT_BIF := $(BUILD_DIR)/bootgen.bif

# Package files for xmutil
FPGA_BIN := $(SALIDA)/$(APP).bit.bin
XCL_BIN := $(SALIDA)/$(APP).xclbin
SHELL_JSON := $(SALIDA)/shell.json
DTB_FILE := $(SALIDA)/$(APP).dtbo

# V++ flags used for the profiled hardware build
VPP_FLAGS := --target $(TARGET) --optimize 0 --report_level 0 --save-temps
VPP_LDFLAGS := --clock.id 1:softmax_hls_1.ap_clk --profile.exec all:all

.PHONY: all build clean cleanall
.DELETE_ON_ERROR:

all: build

build: $(FPGA_BIN) $(XCL_BIN) $(SHELL_JSON) $(DTB_FILE)

# Synthesize HLS and export the kernel as an XO
$(HLS_XO): $(HLS_FILES) $(HLS_CONFIG) Makefile | $(BUILD_DIR)
	cp $(HLS_FILES) "$(HLS_CONFIG)" "$(BUILD_DIR)/"
	cd "$(BUILD_DIR)" && v++ -c --mode hls --config ./softmax_config.cfg --work_dir ./hls_build
	cd "$(BUILD_DIR)" && vitis-run --mode hls --package --config ./softmax_config.cfg --work_dir ./hls_build

# Link the kernel against the KV260 platform
$(LINK_OUTPUT): $(HLS_XO) $(PLATFORM_DIR)/kv260_custom.xpfm $(wildcard $(PLATFORM_DIR)/hw/* $(PLATFORM_DIR)/sw/*)
	cp -a "$(PLATFORM_DIR)/." "$(BUILD_DIR)/platform/"
	cd "$(BUILD_DIR)" && v++ --link $(VPP_FLAGS) $(VPP_LDFLAGS) --platform "./$(PLATFORM)" \
	    --input_files ./hls_build/softmax_hls.xo --output "$(APP).xclbin" --temp_dir "./tmp.$(TARGET)"

$(SYSTEM_BIT): $(LINK_OUTPUT)
	cp "$(TEMP_DIR)/link/int/system.bit" "$@"

$(BOOT_BIF): | $(BUILD_DIR)
	printf 'all:{system.bit}\n' > "$@"

# Package the hardware files
$(FPGA_BIN): $(SYSTEM_BIT) $(BOOT_BIF) | $(SALIDA)
	cd "$(BUILD_DIR)" && bootgen -w -arch zynqmp -process_bitstream bin -image bootgen.bif
	cp "$(BUILD_DIR)/system.bit.bin" "$@"

$(XCL_BIN): $(LINK_OUTPUT) | $(SALIDA)
	cp "$<" "$@"

$(SHELL_JSON): $(PLATFORM_DIR)/shell.json | $(SALIDA)
	cp "$<" "$@"

$(DTB_FILE): $(PLATFORM_DIR)/pl.dtbo | $(SALIDA)
	cp "$<" "$@"

$(BUILD_DIR) $(SALIDA):
	mkdir -p "$@"

# Cleaning rules
clean:
	$(RM) -r -- "$(BUILD_DIR)"

cleanall: clean
	$(RM) -- "$(FPGA_BIN)" "$(XCL_BIN)" "$(SHELL_JSON)" "$(DTB_FILE)"