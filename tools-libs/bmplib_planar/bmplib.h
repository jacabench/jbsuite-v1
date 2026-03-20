#ifndef BMPLIB_H
#define BMPLIB_H

#include <stdint.h>

// Macro: Calculate index for Planar Buffers
// k = channel (0:R, 1:G, 2:B), i = row, j = col, w = width, h = height
#define IDX_PLANAR(k, i, j, w, h) ((k) * (h) * (w) + (i) * (w) + (j))

#pragma pack(push, 1)
typedef struct {
    unsigned short signature;
    unsigned int fileSize;
    unsigned short reserved1;
    unsigned short reserved2;
    unsigned int dataOffset;
    unsigned int headerSize;
    int width;
    int height;
    unsigned short planes;
    unsigned short bitsPerPixel;
    unsigned int compression;
    unsigned int imageSize;
    int xPixelsPerMeter;
    int yPixelsPerMeter;
    unsigned int colorsUsed;
    unsigned int colorsImportant;
} BMPHeader;
#pragma pack(pop)

/**
 * @brief Reads a BMP file (8 or 24 bit) and returns planar data.
 *
 * @param path File path.
 * @param bmpHeader Pointer to header struct to be filled.
 * @return Pointer to buffer (RRR...GGG...BBB...). Returns NULL on error.
 * Caller must free() memory.
 */
unsigned char* getBMPImage(const char* path, BMPHeader* bmpHeader);


/**
 * @brief Lê um BMP e escreve diretamente num buffer pré-alocado.
 *
 * @param path Caminho do arquivo.
 * @param dest_buffer Ponteiro para o buffer estático onde os dados serão gravados.
 * @param max_size Tamanho total do buffer (para evitar overflow).
 * @param bmpHeader Header para preencher as dimensões.
 * @return 0 em sucesso, -1 em erro.
 */
int loadBMPStatic(const char* path, unsigned char* dest_buffer, int max_size, BMPHeader* bmpHeader);

/**
 * @brief Saves a planar buffer as a 24-bit BMP file.
 *
 * @param path Output file path.
 * @param planar_data Input buffer (Planar format).
 * @param header Header containing dimensions.
 * @return 0 on success, -1 on failure.
 */
int setBMPImage(const char* path, const unsigned char* planar_data, BMPHeader* header);

int hasBMPExtension(const char *filename);


#endif
