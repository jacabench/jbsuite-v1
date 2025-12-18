main: main.o bmplib.o
	gcc -o main main.o bmplib.o

main.o: main.c bmplib.h
	gcc -c main.c 

bmplib.o: bmplib.c
	gcc -c bmplib.c -o bmplib.o

clean:
	rm main *.o