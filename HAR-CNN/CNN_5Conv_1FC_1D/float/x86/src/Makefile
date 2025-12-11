conv5: check_dirs conv5.o conv_tb5.o
	clang++ conv5.o conv_tb5.o -o ../bin/conv5 -lm

conv_tb5.o: conv_tb5.cpp conv_tb5.h conv5.h
	clang++ -c conv_tb5.cpp -o conv_tb5.o

conv5.o: conv5.cpp conv5.h
	clang++ -c conv5.cpp -o conv5.o

check_dirs:
	@mkdir -p ../bin

.PHONY: clean
clean:
	rm -f conv5.o conv_tb5.o
	rm -f ../bin/conv5
	rm -f ../bin/output.bin
