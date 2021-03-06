#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include <iostream>
#include <climits>

int main ()
{

  unsigned long long ull_zero = 0x0000000000000000;
  std::cout <<  "ull_zero " << std::dec << ull_zero << std::endl;
  std::cout << "ull_zero " << std::hex << ull_zero << std::endl;
  std::cout <<  "ull_zero-1 " << std::dec << ull_zero -1 << std::endl;
  std::cout << "ull_zero - 1 " << std::hex << ull_zero -1  << std::endl;

  std::cout << std::endl << std::endl;

  std::cout << "ULONG_MAX:" <<std::dec  << ULONG_MAX << std::endl;

  unsigned long long ull_max = 0xFFFFFFFFFFFFFFFF;
  std::cout << "ull_max: " << std::dec << ull_max << std::endl;
  std::cout << "ull_max: " << std::hex << ull_max << std::endl;
  std::cout << "ull_max+1: " << std::dec << ull_max+1 << std::endl;
  std::cout << "ull_max+1: " << std::hex << ull_max+1 << std::endl;

  std::cout << std::endl << std::endl;


  std::cout << "LLONG_MIN:" <<std::hex  << LLONG_MIN << std::endl;
  long long ll_min = 0x8000000000000000;
  std::cout << "ll_min:" <<std::dec  << ll_min << std::endl;
  std::cout << "ll_min:"<< std::hex << ll_min  << std::endl;
  std::cout << "ll_min-1: " <<std::dec  << ll_min - 1 << std::endl;
  std::cout << "ll_min-1: " << std::hex << ll_min - 1 << std::endl;

  std::cout << std::endl << std::endl;

  std::cout << "LLONG_MAX:" <<std::hex  << LLONG_MAX << std::endl;
  long long ll_max = 0x7FFFFFFFFFFFFFFF;
  std::cout << "ll_max:" <<std::dec  << ll_max << std::endl;
  std::cout << "ll_max:"<< std::hex << ll_max  << std::endl;
  std::cout << "ll_max+1: " <<std::dec  << ll_max + 1 << std::endl;
  std::cout << "ll_max+1: " << std::hex << ll_max + 1 << std::endl;

  std::cout << std::endl << std::endl;

  unsigned long long ull_negative_1  = -1;
  long long ll_negative_1  = -1;
  std::cout << "ull_negative_1:" <<std::hex  << ull_negative_1 << std::endl;
  std::cout << "ll_negative_1:" <<std::hex  << ll_negative_1 << std::endl;
  std::cout << "test ull_negative_1 == ll_negative_1 : " << std::hex  << (ull_negative_1 == ll_negative_1) << std::endl;
  std::cout << "test ll_negative_1 == -1 : " << std::hex  << (ull_negative_1 == -1) << std::endl;
  std::cout << "test ll_negative_1 > 0 : " << std::hex  << (ull_negative_1 > 0) << std::endl;

  return 0;
}

/* output

ull_zero 0
ull_zero 0
ull_zero-1 18446744073709551615
ull_zero - 1 ffffffffffffffff


ULONG_MAX:18446744073709551615
ull_max: 18446744073709551615
ull_max: ffffffffffffffff
ull_max+1: 0
ull_max+1: 0


LLONG_MIN:8000000000000000
ll_min:-9223372036854775808
ll_min:8000000000000000
ll_min-1: 9223372036854775807
ll_min-1: 7fffffffffffffff


LLONG_MAX:7fffffffffffffff
ll_max:9223372036854775807
ll_max:7fffffffffffffff
ll_max+1: -9223372036854775808
ll_max+1: 8000000000000000


ull_negative_1:ffffffffffffffff
ll_negative_1:ffffffffffffffff
test ull_negative_1 == ll_negative_1 : 1
test ll_negative_1 == -1 : 1
test ll_negative_1 > 0 : 1
 * *
 *
 *
 */

