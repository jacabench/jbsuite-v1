
# JAVABenchSuite
# smooth kernel

# the following definitions are passed through gcc parameters
#define CONFIG 1
#define CHECK_VALIDITY 0 or 1
#define TIMING 0 or 1
#define READ 1..4
#define RUN_OPTION 1..5 // see below

# READ options:
#define VIA_ASCII_FILES 1
#define VIA_INIT_ARRAYS 2 // requires #include from file: see main.c
#define VIA_LOOP 3
#define VIA_BMP_FILES 4

#/*	Code versions
#*	RUN_OPTION:
#*	1: smooth
#*	2: smooth_reuse1
#*	3: smooth_reuse2
#*	4: smooth_reuse3
#*	5: smooth_reuse4

# the path to the JACABenchSuite: JBSPATH is a system variable
ifeq ($(OS),Windows_NT)
	JACAPATH ?= $(subst \,\\,$(JBSPATH))
else
	JACAPATH ?= $(JBSPATH)
endif

#variables that can be overridden
CONFIG ?= 1
CHECK_VALIDITY ?= 0
TIMING ?= 1
READ ?= 2
RUN_OPTION ?= 1

#compiler optimizations
OPT ?= -O3 #-march=native

# to work in Windows and Linux
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
	PATH_SEP1 = /
endif

JACABENCH = $(JACAPATH)$(PATH_SEP)benchs$(PATH_SEP)smooth
JACABENCH1 = $(JACABENCH)$(PATH_SEP)smooth-img-array
JACASOURCES = $(JACABENCH1)$(PATH_SEP)src

JACAINCLUDES = $(JACAPATH)$(PATH_SEP)includes
JACALIBS = $(JACAPATH)$(PATH_SEP)tools-libs$(PATH_SEP)bmplib

INPUT_PATH = $(JACAPATH)$(PATH_SEP1)benchs$(PATH_SEP1)smooth$(PATH_SEP1)images
OUTPUT_PATH =$(JACAPATH)$(PATH_SEP1)benchs$(PATH_SEP1)smooth$(PATH_SEP1)smooth-img-array$(PATH_SEP1)output
GOLD_PATH = $(JACAPATH)$(PATH_SEP1)benchs$(PATH_SEP1)smooth$(PATH_SEP1)gold

echo1:
	@echo JACABench path: $(JACAPATH)
	@echo Sources: $(JACASOURCES)
	@echo Includes: $(JACAINCLUDES)
	@echo Images: $(INPUT_PATH)
	@echo Output folder: $(OUTPUT_PATH)
	@echo Gold folder: $(GOLD_PATH)

SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin
PROGNAME = smooth$(EXE)
TARGET = $(BIN_DIR)$(PATH_SEP)$(PROGNAME)

CC = gcc
CFLAGS = $(OPT) \
	-I$(JACAINCLUDES) \
	-I$(INPUT_PATH) \
	-I$(GOLD_PATH) \
	-Wall -Wextra -std=gnu99 \
	-D RUN_OPTION=$(RUN_OPTION) \
	-D CONFIG=$(CONFIG) \
	-D TIMING=$(TIMING) \
	-D READ=$(READ) \
	-D CHECK_VALIDITY=$(CHECK_VALIDITY)

LDFLAGS = -lm

SRCS = $(SRC_DIR)$(PATH_SEP)smooth.c \
		$(SRC_DIR)$(PATH_SEP)utils.c \
		$(JACAINCLUDES)$(PATH_SEP)timer.c \
		$(JACAINCLUDES)$(PATH_SEP)IO.c \
		$(JACAINCLUDES)$(PATH_SEP)utilsjacabenchs.c \
		$(SRC_DIR)$(PATH_SEP)main.c

# -------- Scenario read options --------
ifeq ($(READ),1)
	CFLAGS += -D INPUT_PATH='$(INPUT_PATH)'
	CFLAGS += -D OUTPUT_PATH='$(OUTPUT_PATH)'
	CFLAGS += -D GOLD_PATH='$(GOLD_PATH)'
else ifeq ($(READ),4)
	CFLAGS += -D INPUT_PATH='$(INPUT_PATH)'
	CFLAGS += -D OUTPUT_PATH='$(OUTPUT_PATH)'
	SRCS +=	$(JACALIBS)$(PATH_SEP)bmplib.c
	CFLAGS += -I$(JACALIBS)
endif

# Search paths for prerequisites so pattern rule can find %.c
VPATH := $(SRC_DIR):$(JACAINCLUDES):$(JACALIBS)

# Place all object files directly inside build
OBJS := $(addprefix $(OBJ_DIR)$(PATH_SEP),$(notdir $(SRCS:.c=.o)))
OBJS1 := $(addprefix $(OBJ_DIR)$(PATH_SEP1),$(notdir $(SRCS:.c=.o)))

echo2:
	echo OBJS $(OBJS)
	echo OBJS1 $(OBJS1)
	echo VPATH $(VPATH)

.PHONY: all clean run

# Default builds the target
all: $(TARGET)

# Link rule for final executable
$(TARGET): $(OBJS) | $(BIN_DIR)
	@echo Generating $(TARGET)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Generic compile rule for any source; compiled files in build/
$(OBJ_DIR)$(PATH_SEP)%.o: %.c | $(OBJ_DIR)
	@echo Generating $(OBJS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean objects; keeps binary
clean:
	@echo Removing $(OBJS1)
	$(RM) $(RMFLAGS) $(OBJS1)

run: $(TARGET)
	@echo Running $(TARGET)...
	$(TARGET)