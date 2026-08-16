#ifndef CASIOLLM_HOST_BFILE_H
#define CASIOLLM_HOST_BFILE_H

#include <stdint.h>

enum { BFile_ReadOnly = 0, BFile_ReadWrite = 1, BFile_File = 0 };

int BFile_Open(uint16_t const *path, int mode);
int BFile_Close(int fd);
int BFile_Size(int fd);
int BFile_Read(int fd, void *buffer, int size, int offset);
int BFile_Write(int fd, void const *buffer, int size);
int BFile_Seek(int fd, int offset);
int BFile_Create(uint16_t const *path, int type, int *size);
int BFile_Remove(uint16_t const *path);

#endif
