# Makefile for har-knn (Linux/Windows)
# 
# Usage:
#   make default
#   make profiling
#   make -B <rule> or make clean && make <rule>
#

#define DT 2 or 1
#define READ 4 or 2
#define SPECIALIZED 1 or 0
#define MATH_TYPE 1
#define DIST_METHOD 1
#define USE_SQRT 1 or 0
#define DIMEM 0
#define VERIFY 0
#define STREAMING 1
#define K 3
#define TIMMING 0
#define NORMALIZE 1
#define SCENARIO_FEATURES 1 or 2
#define NUM_FEATURES 43 or 90
#define UPDATE 1 or 2
#define NUM_THREADS 1..N

ifndef JBSPATH
$(error Environment variable JBSPATH is not set)
endif

# -------- JACABench root (from JBSPATH) --------
ifeq ($(OS),Windows_NT)
	JACAPATH ?= $(subst \,\\,$(JBSPATH))
else
	JACAPATH ?= $(JBSPATH)
endif

# -------- Scenario build-time defines --------
#variable can be overridden: wisdm or papmap2
SCENARIO ?= wisdm
K ?= 3
DT ?= 2
READ ?= 4
SPECIALIZED ?= 0
USE_SQRT ?= 0
MATH_TYPE ?= 1
DIST_METHOD ?= 1
UPDATE ?= 1
NUM_THREADS ?= 1
TIMING ?= 1

# -------- Compiler optimizations --------
OPT ?= -O3

# -------- Platform portability --------
ifeq ($(OS),Windows_NT)
	RM = del
	RMFLAGS = /Q
	EXE = .exe
	PATH_SEP = /
	PATH_SEP1 = \\
else
	RM = rm
	RMFLAGS = -f
	EXE =
	PATH_SEP = /
	PATH_SEP = /
endif

JACASOURCES = $(JACAPATH)$(PATH_SEP)benchs

# -------- Scenario build-time defines --------
ifeq ($(SCENARIO),wisdm)
INCLUDES = $(JACASOURCES)$(PATH_SEP)har-knn$(PATH_SEP)scenario-wisdm
SCENARIO_FEATURES ?= 1
INPUT_DATA = $(JACAPATH)$(PATH_SEP1)benchs$(PATH_SEP1)har-knn$(PATH_SEP1)scenario-wisdm
else
INCLUDES = $(JACASOURCES)$(PATH_SEP)har-knn$(PATH_SEP)scenario-pamap2
SCENARIO_FEATURES ?= 2
INPUT_DATA = $(JACAPATH)$(PATH_SEP1)benchs$(PATH_SEP1)har-knn$(PATH_SEP1)scenario-pamap2
endif

JACAINCLUDES = $(JBSPATH)$(PATH_SEP)includes

echo1:
	@echo JAVABench root: $(JACAPATH)
	@echo Benchs: $(JACASOURCES)
	@echo Data folder: $(INCLUDES)
	@echo Includes: $(JACAINCLUDES)

SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin
RUN_DIR = output
PROGNAME = harknn$(EXE)
TARGET = $(BIN_DIR)$(PATH_SEP)$(PROGNAME)

CC = gcc
CFLAGS = $(OPT) \
	-I$(INCLUDES) \
	-I$(JACAINCLUDES) \
	-Wall -Wextra -std=gnu99 \
	-fopenmp \
	-D NUM_THREADS=$(NUM_THREADS) \
	-D K=$(K) \
	-D UPDATE=$(UPDATE) \
	-D DT=$(DT) \
	-D SPECIALIZED=$(SPECIALIZED) \
	-D MATH_TYPE=$(MATH_TYPE) \
	-D DIST_METHOD=$(DIST_METHOD) \
	-D USE_SQRT=$(USE_SQRT) \
	-D SCENARIO_FEATURES=$(SCENARIO_FEATURES) \
	-D TIMING=$(TIMING) \
	-D READ=$(READ) -D DIMEM=0 -D VERIFY=0 -D STREAMING=1 -D NORMALIZE=1

# -------- Scenario read options --------
ifeq ($(READ),3)
TRAIN_PATH = '$(INPUT_DATA)$(PATH_SEP1)train2_notnorm.dat'
TEST_PATH = '$(INPUT_DATA)$(PATH_SEP1)test2_notnorm.dat'
CFLAGS += -D TRAIN_PATH=\"$(TRAIN_PATH)\"
CFLAGS += -D TEST_PATH=\"$(TEST_PATH)\"
else ifeq ($(READ),4)
TRAIN_PATH = '$(INPUT_DATA)$(PATH_SEP1)train2_norm.dat'
TEST_PATH = '$(INPUT_DATA)$(PATH_SEP1)test2_notnorm.dat'
CFLAGS += -D TRAIN_PATH=\"$(TRAIN_PATH)\"
CFLAGS += -D TEST_PATH=\"$(TEST_PATH)\"
endif	

LDFLAGS = -lm

PROFILING_FLAGS = -g -pg
PROFILING = gprof -b

SRCS = $(SRC_DIR)$(PATH_SEP)knn.c \
    $(SRC_DIR)$(PATH_SEP)utils.c \
    $(SRC_DIR)$(PATH_SEP)io.c \
    $(SRC_DIR)$(PATH_SEP)features.c \
    $(JACAINCLUDES)$(PATH_SEP)timer.c \
	$(SRC_DIR)$(PATH_SEP)main.c

# Search paths for prerequisites so pattern rule can find %.c
VPATH := $(SRC_DIR):$(JACAINCLUDES)

# Place all object files directly inside build
OBJS := $(addprefix $(OBJ_DIR)$(PATH_SEP),$(notdir $(SRCS:.c=.o)))
OBJS1 := $(addprefix $(OBJ_DIR)$(PATH_SEP1),$(notdir $(SRCS:.c=.o)))

.PHONY: default profiling clean cleanall run runprofiling

# Default builds the target
default: $(TARGET)

# Add profiling flags
profiling: CFLAGS += $(PROFILING_FLAGS)
profiling: $(TARGET)

# Link rule for final executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Generic compile rule for any source; compiled files in build/
$(OBJ_DIR)$(PATH_SEP)%.o: %.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean objects; keeps binary
clean:
	@echo Removing $(OBJS1)
	$(RM) $(RMFLAGS) $(OBJS1)

# Clean everything (binary + profiling data)
cleanall: clean
	@echo Removing $(OBJS1) and $(BIN_DIR)/gmon.out...
	$(RM) $(RMFLAGS) $(TARGET) $(BIN_DIR)/gmon.out

run: $(TARGET)
	@echo Running $(TARGET)...
	$(TARGET)

runprofiling: $(TARGET)
	cd $(RUN_DIR) && \
	../$(BIN_DIR)/$(PROGNAME) && \
	$(PROFILING) ../$(BIN_DIR)/$(PROGNAME)