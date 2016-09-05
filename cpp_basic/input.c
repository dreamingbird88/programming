/*

gcc -o input input.c
./input x y z

*/
#include <stdio.h>

int main(int argc, char**argv) {
  printf("Total input parameters = %d\n", argc);
  for (int i = 0; i < argc; ++i) {
    printf("arg[%d] = %s\n", i, argv[i]);
  }
}
