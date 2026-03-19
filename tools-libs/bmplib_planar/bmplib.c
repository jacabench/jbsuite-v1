#include "bmplib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned char* getBMPImage(const char* path, BMPHeader* bmpHeader) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "ERRO: Nao foi possivel abrir o ficheiro '%s'.\n", path);
        return NULL;
    }

    if (fread(bmpHeader, sizeof(BMPHeader), 1, file) != 1) {
        fprintf(stderr, "ERRO: Falha ao ler o cabecalho do BMP.\n");
        fclose(file);
        return NULL;
    }

    // Validações
    if (bmpHeader->signature != 0x4D42 || bmpHeader->compression != 0 ||
        (bmpHeader->bitsPerPixel != 8 && bmpHeader->bitsPerPixel != 24)) {
        fprintf(stderr, "ERRO: Formato de BMP nao suportado (suporta apenas 8/24 bits sem compressao).\n");
        fclose(file);
        return NULL;
    }

    int width = bmpHeader->width;
    int height = abs(bmpHeader->height);
    int plane_size = width * height;
    
    // Aloca memória para os 3 canais planares
    unsigned char* planar_data = (unsigned char*)malloc(plane_size * 3);
    if (!planar_data) {
        fprintf(stderr, "ERRO: Falha na alocacao de memoria para a imagem planar.\n");
        fclose(file);
        return NULL;
    }
    
    // Apontadores para o início de cada plano de cor
    unsigned char* r_plane = planar_data;
    unsigned char* g_plane = planar_data + plane_size;
    unsigned char* b_plane = planar_data + 2 * plane_size;

    fseek(file, bmpHeader->dataOffset, SEEK_SET);

    if (bmpHeader->bitsPerPixel == 24) {
        int row_padding = (4 - (width * 3) % 4) % 4;
        unsigned char bgr[3];

        for (int i = height - 1; i >= 0; i--) { // BMPs são lidos de baixo para cima
            for (int j = 0; j < width; j++) {
                fread(bgr, 3, 1, file);
                int idx = i * width + j;
                // Lê BGR do ficheiro e escreve nos planos R, G, B separados
                r_plane[idx] = bgr[2];
                g_plane[idx] = bgr[1];
                b_plane[idx] = bgr[0];
            }
            fseek(file, row_padding, SEEK_CUR);
        }
    } else { // 8-bit (escala de cinza)
        int row_padding = (4 - width % 4) % 4;
        unsigned char gray_value;

        for (int i = height - 1; i >= 0; i--) {
            for (int j = 0; j < width; j++) {
                fread(&gray_value, 1, 1, file);
                int idx = i * width + j;
                // Converte o valor de cinza para RGB
                r_plane[idx] = gray_value;
                g_plane[idx] = gray_value;
                b_plane[idx] = gray_value;
            }
            fseek(file, row_padding, SEEK_CUR);
        }
    }

    fclose(file);
    return planar_data;
}


int setBMPImage(const char* path, const unsigned char* planar_data, BMPHeader* header) {
    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "ERRO: Nao foi possivel criar o ficheiro '%s'.\n", path);
        return -1;
    }

    int width = header->width;
    int height = abs(header->height);
    int row_padding = (4 - (width * 3) % 4) % 4;
    int plane_size = width * height;

    // Cria um novo cabeçalho para garantir que os valores estão corretos para um BMP de 24 bits
    BMPHeader newHeader = *header;
    newHeader.signature = 0x4D42;
    newHeader.dataOffset = sizeof(BMPHeader);
    newHeader.bitsPerPixel = 24;
    newHeader.compression = 0;
    newHeader.imageSize = (width * 3 + row_padding) * height;
    newHeader.fileSize = newHeader.dataOffset + newHeader.imageSize;

    fwrite(&newHeader, sizeof(BMPHeader), 1, file);

    const unsigned char* r_plane = planar_data;
    const unsigned char* g_plane = planar_data + plane_size;
    const unsigned char* b_plane = planar_data + 2 * plane_size;

    unsigned char padding_byte = 0;

    for (int i = height - 1; i >= 0; i--) { // BMPs são escritos de baixo para cima
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            unsigned char bgr[3];
            // Lê dos planos R, G, B e escreve no formato BGR
            bgr[2] = r_plane[idx];
            bgr[1] = g_plane[idx];
            bgr[0] = b_plane[idx];
            fwrite(bgr, 3, 1, file);
        }
        fwrite(&padding_byte, 1, row_padding, file);
    }

    fclose(file);
    return 0;
}
