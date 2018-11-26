#include "Exercise.hpp"
#include <sstream>

int main(int ac, char**av) {
  if( ac == 2 ) {
    unsigned int size = 16;
    std::istringstream iss(av[1]);
    iss>>size;
    if( size < 16 ) size = 16;
    else if ( size > 32 ) size = 32; 
    Exercise e(1<<size);
    e.run();
  }
  else {
    Exercise e;
    e.run();
  }
  return 0;
}
