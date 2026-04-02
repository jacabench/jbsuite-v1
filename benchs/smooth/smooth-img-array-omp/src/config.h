/**
*       JACABednch: JACA Benchmark Suite
*       smooth kernel
*	File with the main configuration options.
*
*/
#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

#define W 3 // window is of size W x W: fixed

typedef uint8_t uint8;

/*	Code versions
*	RUN_OPTION:
*	1: smooth
*	2: smooth_reuse1
*	3: smooth_reuse2
*	4: smooth_reuse3
*	5: smooth_reuse4
*/
#ifndef RUN_OPTION
#define RUN_OPTION 1
#endif

/*
*	Check the validity of the output results
*	CHECK_VALIDITY:
*	1: checksum
*	2: output image compared to gold output
*/
#ifndef CHECK_VALIDITY
#define CHECK_VALIDITY 1
#endif

/*
*	The options regarding the input image
*/
#define VIA_ASCII_FILES 1
#define VIA_INIT_ARRAYS 2  // requires #include from file: see main.c
#define VIA_LOOP 3
#define VIA_BMP_FILES 4

#ifndef READ
#define READ VIA_LOOP
#endif

/*	Configurations for input image sizes
*
*	CONFIG:
*	1: SMALL: 320x240: QVGA
*	2: MEDIUM: 640x480: VGA
*	3: LARGE1: 1024x768: XGA / EVGA
*	4: LARGE2: 1280x720: HD
*	5: EXTRA_LARGE: 1920x1080: Full HD
*	6: EXTRA_EXTRA_LARGE: 3840x2160: 4K Digital Cinema
*/
#ifndef CONFIG
#define CONFIG 1
#endif

#if CONFIG == 1 //qvga
	#define sizeX 320
	#define sizeY 240
#elif CONFIG == 2 //vga
	#define sizeX 640
	#define sizeY 480
#elif CONFIG == 3 //xga
	#define sizeX 1024
	#define sizeY 768
#elif CONFIG == 4 //hd
	#define sizeX 1280
	#define sizeY 720
#elif CONFIG == 5  //fullHD
	#define sizeX 1920
	#define sizeY 1080
#elif CONFIG == 6 //4k
	#define sizeX 3840
	#define sizeY 2160
#endif

#define STR_IMPL(A) #A
#define STR(A) STR_IMPL(A)

#if READ == VIA_ASCII_FILES  // for now it considers only QVGA , CONFIG = 1
    #include "IO.h"

	#define INPUT_FILE input.dsp
	#define OUTPUT_FILE output.dsp
	//#define INPUT_DIRECTORY images
	//#define OUTPUT_DIRECTORY output
	//#define GOLD_DIRECTORY gold

	#if defined(WIN64) || defined(_WIN64) || defined(WIN32) || defined(_WIN32) || defined(__CYGWIN__)
	#define INPUT_RESOURCE STR(INPUT_PATH\\INPUT_FILE)
	#define OUTPUT_RESOURCE STR(OUTPUT_PATH\\OUTPUT_FILE)
	#define OUTPUT_GOLD_RESOURCE STR(GOLD_PATH\\OUTPUT_FILE)
	#else
	#define INPUT_RESOURCE STR(INPUT_PATH/INPUT_FILE)
	#define OUTPUT_RESOURCE STR(OUTPUT_PATH/OUTPUT_FILE)
	#define OUTPUT_GOLD_RESOURCE STR(GOLD_PATH/OUTPUT_FILE)
	#endif

#elif READ == VIA_BMP_FILES
	#include "bmplib.h"

	#if CONFIG == 1 //qvga
	#define INPUT_FILE qvga.bmp
	#elif CONFIG == 2 //vga
	#define INPUT_FILE vga.bmp
	#elif CONFIG == 3 //xga
	#define INPUT_FILE xga.bmp
	#elif CONFIG == 4 //hd
	#define INPUT_FILE hd.bmp
	#elif CONFIG == 5  //fullHD
	#define INPUT_FILE fullHD.bmp
	#elif CONFIG == 6 //4k
	#define INPUT_FILE 4k.bmp
	#endif
	
	//#define INPUT_DIRECTORY images
	//#define OUTPUT_DIRECTORY output
	#define OUTPUT_FILE INPUT_FILE

	#if defined(WIN64) || defined(_WIN64) || defined(WIN32) || defined(_WIN32) || defined(__CYGWIN__)
	#define INPUT_RESOURCE STR(INPUT_PATH\\INPUT_FILE)
	#define OUTPUT_RESOURCE STR(OUTPUT_PATH\\OUTPUT_FILE)
	#else
	#define INPUT_RESOURCE STR(INPUT_PATH/INPUT_FILE)
	#define OUTPUT_RESOURCE STR(OUTPUT_PATH/OUTPUT_FILE)
	#endif

#endif

#endif