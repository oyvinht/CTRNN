#include <iostream>
#include "CTRNN.h"

int main() {
    
  ctrnn::CTRNN net(2, 0.01);

  std::cout << "Activation " << net.getActivation(0) << std::endl;
  
  return 0;
  
}
