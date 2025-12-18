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
/*
* João MP Cardoso, Dec. 2023
* Added file_name parameter to functions
* Added option for unsigned char
* Changed input/output calls to read/write from/to files from the beginning. 
*/

#include <stdio.h>
#include <stdlib.h>

int input_dsp (dest, words, data_type, file_name)
void *dest;
int  words;
int  data_type;
char *file_name;
{
	
  FILE *input_fp=NULL;

  int success;
  int		i;
  double	*float_ptr;
  int		*int_ptr;
  unsigned	*unsigned_ptr;
  unsigned char	*unsigned_char_ptr;

  if ((input_fp = fopen (file_name, "r")) == NULL) {
    printf ("** Error: cannot open %s.\n", file_name);
    exit(1);
  }

  if (words <= 0) {
    printf ("** Error: trying to read a negative or zero number of words.\n");
    exit (1);
  }

  switch (data_type) {
    case 0:
      float_ptr = (double *) dest;
      for (i = 0; i < words; i++)
        success = fscanf (input_fp, "%lf", &float_ptr[i]);
      break;
    case 1:
      int_ptr = (int *) dest;
      for (i = 0; i < words; i++) {
        success = fscanf (input_fp, "%d", &int_ptr[i]);
		//printf("%d, ",input_fp);
	  }
      break;
    case 2:
      unsigned_ptr = (unsigned *) dest;
      for (i = 0; i < words; i++) {
        success = fscanf (input_fp, "%u", &unsigned_ptr[i]);
	  }
      break;
    case 3:
      success = fread( (char *) dest, 1, words, input_fp);
      break;
	case 4:
      unsigned_char_ptr = (unsigned char *) dest;
	  unsigned int a;
      for (i = 0; i < words; i++) {
        success = fscanf (input_fp, "%u", &a);
		unsigned_char_ptr[i] = (unsigned char) a;
		//printf("%d \n", unsigned_char_ptr[i]);
	  }
	  
      break;
    default:
      printf ("** Error: trying to use an undefined data type. \n");
      exit(1);
      break;
  }

  return success;
}


void output_dsp (src, words, data_type, file_name)
void *src;
int  words;
int  data_type;
char *file_name;
{
	
  FILE *output_fp=NULL;

  int i;
  double *float_ptr;
  int *int_ptr;
  unsigned *unsigned_ptr;
  unsigned char *unsigned_char_ptr;

  if ((output_fp = fopen (file_name, "w")) == NULL) {
    printf ("** Error: cannot open %s.\n", file_name);
    exit(1);
  }

  if (words <= 0) {
    printf ("** Error: trying to write a negative or zero number of words.\n");
    exit (1);
  }

  switch (data_type) {
    case 0:
      float_ptr = (double *) src;
      for (i = 0; i < words; i++)
        fprintf (output_fp, "%lf\n", float_ptr[i]);
      break;
    case 1:
      int_ptr = (int *) src;
      for (i = 0; i < words; i++)
        fprintf (output_fp, "%d\n", int_ptr[i]);
      break;
    case 2:
      unsigned_ptr = (unsigned *) src;
      for (i = 0; i < words; i++)
        fprintf (output_fp, "%u\n", unsigned_ptr[i]);
      break;
    case 3:
      fwrite( (char *) src, 1, words, output_fp);
      break;
	case 4:
      unsigned_char_ptr = (unsigned char *) src;
      for (i = 0; i < words; i++) {
        fprintf (output_fp, "%u\n", unsigned_char_ptr[i]);
		//printf("%u \n", unsigned_char_ptr[i]);
	  }
      break;
    default:
      printf ("** Error: trying to use an undefined data type. \n");
      exit(1);
      break;
  }
  
}
