
#include <stdio.h>
#include <sys/time.h>

double rtclock_ms() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    
	return (Tp.tv_sec*1.0e3 + Tp.tv_usec*1.0e-3);
}

double rtclock_sec() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    
	return (Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


int compare_unsigned_char(unsigned char *input1, unsigned char *input2, int num_elements) {
	int i;
	int success = 1;
	//printf("address: %d %d\n", &input1[0], &input2[0]);
	for(i=0; i<num_elements; i++) {
		if(input1[i] != input2[i]) {
			//if(input2[i] != 0) 
			//printf("%d %d\n", input1[i], input2[i]);
			success = 0;
		}
	}
	return success;
}