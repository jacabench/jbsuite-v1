conv3: check_dirs conv3.o conv_tb3.o
	clang++ conv3.o conv_tb3.o -o ../bin/conv3 -lm

conv_tb3.o: conv_tb3.cpp conv_tb3.h conv3.h
	clang++ -c conv_tb3.cpp -o conv_tb3.o

conv3.o: conv3.cpp conv3.h
	clang++ -c conv3.cpp -o conv3.o

check_dirs:
	@mkdir -p ../bin

.PHONY: clean
clean:
	rm -f conv3.o conv_tb3.o
	rm -f ../bin/conv3
	rm -f ../bin/output.bin
