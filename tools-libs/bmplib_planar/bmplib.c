#include "bmplib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned char* getBMPImage(const char* path, BMPHeader* bmpHeader) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "ERROR: Could not open file '%s'.\n", path);
        return NULL;
    }

    if (fread(bmpHeader, sizeof(BMPHeader), 1, file) != 1) {
        fprintf(stderr, "ERROR: Failed to read BMP header.\n");
        fclose(file);
        return NULL;
    }

    // Validation
    if (bmpHeader->signature != 0x4D42 || bmpHeader->compression != 0 ||
        (bmpHeader->bitsPerPixel != 8 && bmpHeader->bitsPerPixel != 24)) {
        fprintf(stderr, "ERROR: Unsupported BMP format (only 8/24-bit uncompressed).\n");
        fclose(file);
        return NULL;
    }

    int width = bmpHeader->width;
    int height = abs(bmpHeader->height);
    int plane_size = width * height;
    
    // Allocate memory for 3 planar channels
    unsigned char* planar_data = (unsigned char*)malloc(plane_size * 3);
    if (!planar_data) {
        fprintf(stderr, "ERROR: Memory allocation failed.\n");
        fclose(file);
        return NULL;
    }

    // Jump to pixel data
    fseek(file, bmpHeader->dataOffset, SEEK_SET);

    // Calculate row padding (rows are aligned to 4 bytes)
    int row_padding = (4 - (width * (bmpHeader->bitsPerPixel / 8)) % 4) % 4;
    unsigned char padding_byte;

    // Pointers to R, G, B planes
    unsigned char* r_plane = planar_data;
    unsigned char* g_plane = planar_data + plane_size;
    unsigned char* b_plane = planar_data + 2 * plane_size;

    // Read loop (BMP stores bottom-up)
    for (int i = height - 1; i >= 0; i--) {
        for (int j = 0; j < width; j++) {
            unsigned char bgr[3];
            
            if (bmpHeader->bitsPerPixel == 24) {
                fread(bgr, 3, 1, file);
            } else {
                // 8-bit grayscale handling
                unsigned char gray;
                fread(&gray, 1, 1, file);
                bgr[0] = bgr[1] = bgr[2] = gray;
            }

            // Convert Packed BGR -> Planar RGB
            int idx = i * width + j;
            r_plane[idx] = bgr[2]; // R
            g_plane[idx] = bgr[1]; // G
            b_plane[idx] = bgr[0]; // B
        }
        // Skip padding
        for (int k = 0; k < row_padding; k++) fread(&padding_byte, 1, 1, file);
    }

    fclose(file);
    return planar_data;
}

int loadBMPStatic(const char* path, unsigned char* dest_buffer, int max_size, BMPHeader* bmpHeader) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "ERRO: Nao foi possivel abrir '%s'.\n", path);
        return -1;
    }

    if (fread(bmpHeader, sizeof(BMPHeader), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    // Validações básicas
    if (bmpHeader->signature != 0x4D42 || bmpHeader->bitsPerPixel != 24) {
        fprintf(stderr, "ERRO: Formato BMP invalido.\n");
        fclose(file);
        return -1;
    }

    int width = bmpHeader->width;
    int height = abs(bmpHeader->height);
    int required_size = width * height * 3;

    // --- PROTEÇÃO DE MEMÓRIA ESTATICA ---
    if (required_size > max_size) {
        fprintf(stderr, "ERRO: Imagem (%d bytes) maior que o buffer estatico (%d bytes).\n", required_size, max_size);
        fclose(file);
        return -1;
    }

    // Pular para os dados
    fseek(file, bmpHeader->dataOffset, SEEK_SET);

    int row_padding = (4 - (width * 3) % 4) % 4;
    unsigned char padding_byte;

    // Configurar ponteiros para escrever PLANAR (RRR...GGG...BBB) dentro do buffer estático
    int plane_size = width * height;
    unsigned char* r_plane = dest_buffer;
    unsigned char* g_plane = dest_buffer + plane_size;
    unsigned char* b_plane = dest_buffer + 2 * plane_size;

    for (int i = height - 1; i >= 0; i--) {
        for (int j = 0; j < width; j++) {
            unsigned char bgr[3];
            fread(bgr, 3, 1, file); // Lê B, G, R do arquivo

            // Escreve direto no buffer estático destino
            int idx = i * width + j;
            r_plane[idx] = bgr[2]; 
            g_plane[idx] = bgr[1];
            b_plane[idx] = bgr[0];
        }
        // Pular padding do arquivo
        for (int k = 0; k < row_padding; k++) fread(&padding_byte, 1, 1, file);
    }

    fclose(file);
    return 0; // Sucesso
}

int setBMPImage(const char* path, const unsigned char* planar_data, BMPHeader* header) {
    FILE *file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "ERROR: Could not create file '%s'.\n", path);
        return -1;
    }

    int width = header->width;
    int height = abs(header->height);
    int plane_size = width * height;
    int row_padding = (4 - (width * 3) % 4) % 4;

    // Construct new header
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

    // Write loop (Bottom-up)
    for (int i = height - 1; i >= 0; i--) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            unsigned char bgr[3];
            
            // Planar RGB -> Packed BGR
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

// Função auxiliar para verificar se um nome de arquivo termina com ".bmp"
int hasBMPExtension(const char *filename) {
    const char *dot = strrchr(filename, '.'); // Encontra a última ocorrência de '.'
    if (!dot || dot == filename) {
        return 0; // Sem extensão ou o nome começa com '.'
    }
    // Compara a extensão (ignorando maiúsculas/minúsculas para .BMP)
    return strcasecmp(dot, ".bmp") == 0;
}
