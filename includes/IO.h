/*
   Corinna G. Lee
   6 Jun 94: Modified for GNU C version of DSP programs:
 	1. "float" data type converted into "double" data type
	    so that math library functions can be used
	2. add data_type=3 to read/write binary data
   13 Jun 94: Make file pointers static variables to allow multiple calls
	to same I/O routine.  Note: files are not explicitly closed.

   Mark G. Stoodley
   17 Apr 96: caused input_dsp to return an integer that is non-zero if data
		was read from the file
*/

#ifndef IO_H
#define IO_H

int input_dsp (
void *dest,
int  words,
int  data_type,
char *file_name);


void output_dsp (
void *src,
int  words,
int  data_type,
char *file_name);

#endif