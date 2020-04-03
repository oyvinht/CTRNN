#include <iostream>
#include "CTRNN.h"

namespace ctrnn
{
  class CTRNN::impl
  {
  public:
    void init(int netsize)
    {
      this->netsize = netsize;
      activations = new float[netsize];
      biases = new float[netsize];
      externalCurrents = new float[netsize];
      invTimeConstants = new float[netsize];
      potentials = new float[netsize];
      // Init properties of each neuron
      for (int i = 0; i < netsize; i++)
	{
	  activations[i] = 0.0f;
	  biases[i] = 0.0f;
	  externalCurrents[i] = 0.0f;
	  invTimeConstants[i] = 0.0f;
	  potentials[i] = 0.0f;
	}
      // Init weights (netsize * netsize)
      for (int i = 0; i < netsize; i++)
	{
	  for (int j = 0; j < netsize; j++)
	    {
	      weights[i*j] = 0.0f;
	    }
	}
    }
  private:
    float *activations, *biases, *externalCurrents, *invTimeConstants, *potentials;
    float *weights;
    int netsize;
    double stepsize;
  };
  CTRNN::CTRNN(int netsize, double stepsize) : pimpl{std::make_unique<impl>()}
  {
    pimpl->init(netsize);
    return;
  }
  CTRNN::~CTRNN()
  {
    return;
  }
}
