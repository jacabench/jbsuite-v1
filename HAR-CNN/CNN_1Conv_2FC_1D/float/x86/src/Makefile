conv1: check_dirs conv1.o conv_tb1.o
	clang++ conv1.o conv_tb1.o -o ../bin/conv1 -lm

conv_tb1.o: conv_tb1.cpp conv_tb1.h conv1.h
	clang++ -c conv_tb1.cpp -o conv_tb1.o

conv1.o: conv1.cpp conv1.h
	clang++ -c conv1.cpp -o conv1.o

check_dirs:
	@mkdir -p ../bin

.PHONY: clean
clean:
	rm -f conv1.o conv_tb1.o
	rm -f ../bin/conv1
	rm -f ../bin/output.bin
