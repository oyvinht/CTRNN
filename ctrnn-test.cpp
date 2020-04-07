#include <chrono>
#include <fstream>
#include <iostream>
#include "CTRNN.h"


int main() {
  
  ctrnn::CTRNN net(100, 0.01);

  net.setBias(1, 5.0);
  net.setBias(0, -5.0);
  net.setTimeConstant(0, 0.5);
  net.setTimeConstant(1, 0.5);
  net.setWeight(0, 0, 5.0);
  net.setWeight(1, 1, 5.0);
  net.setWeight(1, 0, 10.0);
  net.setWeight(0, 1, -10.0);

  // Check net topology
  std::cout << "Biases:\t" << net.getBias(0) << "\t" << net.getBias(1) << std::endl;
  std::cout << "Time Constants:\t" << net.getTimeConstant(0) << "\t" << net.getTimeConstant(1) << std::endl;
  std::cout << "Weight 0->0:\t" << net.getWeight(0,0) << std::endl;
  std::cout << "Weight 0->1:\t" << net.getWeight(0,1) << std::endl;
  std::cout << "Weight 1->1:\t" << net.getWeight(1,1) << std::endl;
  std::cout << "Weight 1->0:\t" << net.getWeight(1,0) << std::endl;
  std::cout << std::endl << "Neuron activations written to \"net-output.dat\"." << std::endl;

  // Plot outputs
  FILE *outFile = fopen("net-output.dat", "w");
  for (float t = 0; t <= 10; t += 0.01)
    {
      fprintf(outFile, "%1.2f %1.7f %1.7f\n", t, net.getActivation(0), net.getActivation(1));
      net.updatePotentials();
    }
  fclose(outFile);

  // Run heavier CPU test
  int startTime = std::chrono::system_clock::to_time_t ( std::chrono::system_clock::now() );
  for (float t = 0; t <= 1000; t += 0.01)
    {
      net.updatePotentials();
    }
  int stopTime = std::chrono::system_clock::to_time_t ( std::chrono::system_clock::now() );

  std::cout << "Test took " << stopTime - startTime << "s"  << std::endl;
  
  return 0;

}
