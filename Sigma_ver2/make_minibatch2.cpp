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
	hFind = FindFirstFile("C:\\Data_Set\\Tanabe\\cut_data\\0\\*.*", &fd); /* カレントディレクトリのファイルを列挙 */

	if (hFind == INVALID_HANDLE_VALUE) {
		fprintf(stderr, "ファイルを取得できません。\n");
		return 1;
	}

	/* 次々列挙 */
	do {
		a++;
		//ファイル名の表示
		printf("%d:", a);
		printf("%s\n", fd.cFileName);

	} while (FindNextFile(hFind, &fd));
	printf("%d\n", a);
	getchar();
	/* 開放 */
	FindClose(hFind);

	return 0;
} //end of main