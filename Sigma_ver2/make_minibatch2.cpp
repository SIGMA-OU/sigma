#include <iostream>
#include <windows.h>
#include <sstream>
#define dish 4
#define view 4

int main_readfile(void) {
	HANDLE hFind;
	WIN32_FIND_DATA fd;
	char str;
	int a = 0;
	hFind = FindFirstFile("C:\\Data_Set\\Tanabe\\cut_data\\0\\*.*", &fd); /* �J�����g�f�B���N�g���̃t�@�C����� */

	if (hFind == INVALID_HANDLE_VALUE) {
		fprintf(stderr, "�t�@�C�����擾�ł��܂���B\n");
		return 1;
	}

	/* ���X�� */
	do {
		a++;
		//�t�@�C�����̕\��
		printf("%d:", a);
		printf("%s\n", fd.cFileName);

	} while (FindNextFile(hFind, &fd));
	printf("%d\n", a);
	getchar();
	/* �J�� */
	FindClose(hFind);

	return 0;
} //end of main