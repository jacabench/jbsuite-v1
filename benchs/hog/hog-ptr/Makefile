# ==============================================================================
#  Makefile — Projeto hog
#  Suporte: Linux / macOS / Windows (MinGW ou MSYS2)
# ==============================================================================

# ------------------------------------------------------------------------------
#  Detecção de Sistema Operacional
# ------------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
    TARGET_OS := windows
    EXE_EXT   := .exe
    RM        := del /Q /F
    MKDIR     := mkdir
    PATHSEP   := \\
else
    TARGET_OS := unix
    EXE_EXT   :=
    RM        := rm -f
    MKDIR     := mkdir -p
    PATHSEP   := /
endif

# ------------------------------------------------------------------------------
#  Nomes e Diretórios
# ------------------------------------------------------------------------------
TARGET      := hog$(EXE_EXT)

DIR_SRC     := ./src
DIR_BIN     := ./bin
DIR_OBJ     := ./build

BMPLIB_DIR  := ../../../tools-libs/bmplib_planar

# ------------------------------------------------------------------------------
#  Compilador e Flags
# ------------------------------------------------------------------------------
CC          := gcc
CXX         := g++

CFLAGS      := -Wall -Wextra -Wpedantic \
               -O2 \
               -std=c11 \
               -I$(DIR_SRC) \
               -I$(BMPLIB_DIR)

CXXFLAGS    := -Wall -Wextra -Wpedantic \
               -O2 \
               -std=c++17 \
               -I$(DIR_SRC) \
               -I$(BMPLIB_DIR)

LDFLAGS     :=
LDLIBS      := -lm

# ------------------------------------------------------------------------------
#  Fontes e Objetos
# ------------------------------------------------------------------------------

# Fontes do projeto
SRC_C       := $(wildcard $(DIR_SRC)/*.c)
SRC_CXX     := $(wildcard $(DIR_SRC)/*.cpp)

# Fontes da biblioteca externa
LIB_C       := $(wildcard $(BMPLIB_DIR)/*.c)
LIB_CXX     := $(wildcard $(BMPLIB_DIR)/*.cpp)

# Todos os fontes
ALL_SRC_C   := $(SRC_C)   $(LIB_C)
ALL_SRC_CXX := $(SRC_CXX) $(LIB_CXX)

# Objetos: flat em DIR_OBJ — notdir descarta o caminho, mantém só o nome base
OBJ_SRC_C   := $(patsubst %.c,   $(DIR_OBJ)/%.o, $(notdir $(SRC_C)))
OBJ_SRC_CXX := $(patsubst %.cpp, $(DIR_OBJ)/%.o, $(notdir $(SRC_CXX)))
OBJ_LIB_C   := $(patsubst %.c,   $(DIR_OBJ)/%.o, $(notdir $(LIB_C)))
OBJ_LIB_CXX := $(patsubst %.cpp, $(DIR_OBJ)/%.o, $(notdir $(LIB_CXX)))
OBJS        := $(OBJ_SRC_C) $(OBJ_SRC_CXX) $(OBJ_LIB_C) $(OBJ_LIB_CXX)

# Linker: usa g++ se houver qualquer fonte C++, senão gcc
ifneq ($(strip $(ALL_SRC_CXX)),)
    LINKER  := $(CXX)
else
    LINKER  := $(CC)
endif

# ------------------------------------------------------------------------------
#  Regra Padrão
# ------------------------------------------------------------------------------
.PHONY: all
all: $(DIR_BIN)/$(TARGET)

# ------------------------------------------------------------------------------
#  Link Final
# ------------------------------------------------------------------------------
$(DIR_BIN)/$(TARGET): $(OBJS)
	@echo "[LINK]  $@"
	@$(MKDIR) $(DIR_BIN)
	$(LINKER) $(LDFLAGS) $^ -o $@ $(LDLIBS)
	@echo ""
	@echo "  Build concluido: $@"
	@echo ""

# ------------------------------------------------------------------------------
#  Compilação — .c  (regras separadas por origem para o patsubst flat funcionar)
# ------------------------------------------------------------------------------
$(DIR_OBJ)/%.o: $(DIR_SRC)/%.c
	@echo "[CC]    $<"
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

$(DIR_OBJ)/%.o: $(BMPLIB_DIR)/%.c
	@echo "[CC]    $<"
	@$(MKDIR) $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# ------------------------------------------------------------------------------
#  Compilação — .cpp  (regras separadas por origem para o patsubst flat funcionar)
# ------------------------------------------------------------------------------
$(DIR_OBJ)/%.o: $(DIR_SRC)/%.cpp
	@echo "[CXX]   $<"
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(DIR_OBJ)/%.o: $(BMPLIB_DIR)/%.cpp
	@echo "[CXX]   $<"
	@$(MKDIR) $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

# ------------------------------------------------------------------------------
#  Dependências automáticas (geradas com -MMD -MP)
# ------------------------------------------------------------------------------
-include $(OBJS:.o=.d)

# ------------------------------------------------------------------------------
#  Build com símbolos de debug
# ------------------------------------------------------------------------------
.PHONY: debug
debug: CFLAGS   += -g3 -DDEBUG -O0 -fsanitize=address,undefined
debug: CXXFLAGS += -g3 -DDEBUG -O0 -fsanitize=address,undefined
debug: LDFLAGS  += -fsanitize=address,undefined
debug: all

# ------------------------------------------------------------------------------
#  Limpeza
# ------------------------------------------------------------------------------
.PHONY: clean
clean:
	@echo "[CLEAN] Removendo objetos e dependencias..."
	$(RM) $(OBJS) $(OBJS:.o=.d)

.PHONY: distclean
distclean: clean
	@echo "[CLEAN] Removendo executavel e diretorios de build..."
ifeq ($(TARGET_OS),windows)
	@if exist "$(DIR_BIN)\$(TARGET)" $(RM) "$(DIR_BIN)\$(TARGET)"
	@if exist "$(DIR_OBJ)"           rmdir /S /Q "$(DIR_OBJ)"
else
	$(RM) $(DIR_BIN)/$(TARGET)
	$(RM) -r $(DIR_OBJ)
endif

# ------------------------------------------------------------------------------
#  Informações do Build
# ------------------------------------------------------------------------------
.PHONY: info
info:
	@echo "============================================"
	@echo "  Sistema     : $(TARGET_OS)"
	@echo "  Executavel  : $(DIR_BIN)/$(TARGET)"
	@echo "  CC          : $(CC)"
	@echo "  CXX         : $(CXX)"
	@echo "  Linker      : $(LINKER)"
	@echo "  CFLAGS      : $(CFLAGS)"
	@echo "  CXXFLAGS    : $(CXXFLAGS)"
	@echo "  Fontes C    : $(ALL_SRC_C)"
	@echo "  Fontes C++  : $(ALL_SRC_CXX)"
	@echo "============================================"