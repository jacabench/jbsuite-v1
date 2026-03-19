#ifndef BMPLIB_H
#define BMPLIB_H

#include <stdint.h>

// Macro para calcular o índice de um pixel num buffer PLANAR
// k = canal (0:R, 1:G, 2:B), i = linha, j = coluna, w = largura, h = altura
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
 * @brief Lê um ficheiro de imagem BMP (8 ou 24 bits) e retorna os dados em formato planar.
 *
 * @param path O caminho para o ficheiro .bmp.
 * @param bmpHeader Um ponteiro para uma estrutura que será preenchida com o cabeçalho.
 * @return Um ponteiro para um buffer (unsigned char*) com os dados da imagem em
 * formato planar (RRR...GGG...BBB...). Retorna NULL em caso de erro.
 * O chamador é responsável por libertar esta memória com free().
 */
unsigned char* getBMPImage(const char* path, BMPHeader* bmpHeader);

/**
 * @brief Salva um buffer de imagem em formato planar como um ficheiro BMP de 24 bits.
 *
 * @param path O caminho do ficheiro .bmp a ser criado.
 * @param planar_data Um ponteiro para o buffer com os dados da imagem em formato planar.
 * @param header Um ponteiro para a estrutura BMPHeader contendo as dimensões da imagem.
 * @return 0 em caso de sucesso, -1 em caso de falha.
 */
int setBMPImage(const char* path, const unsigned char* planar_data, BMPHeader* header);

#endif // BMPLIB_H