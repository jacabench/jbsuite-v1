TARGET = cnn_trainer
SRC = backprop_main.cpp backprop.cpp

.PHONY: all $(TARGET) clean
all:
	clang++ -O3 -Wall -std=c++11 $(SRC) -o ../bin/$(TARGET) -lm

clean:
	rm -f ../bin/$(TARGET)
