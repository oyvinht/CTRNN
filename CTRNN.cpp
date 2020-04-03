#include <iostream>
#include "CTRNN.h"

namespace ctrnn
{
  struct CTRNN::impl
  {
    float *activations, *biases, *externalCurrents, *invTimeConstants, *potentials;
    float *weights;
    int netsize;
    float stepsize;
    
    void init(int netsize)
    {
      this->netsize = netsize;
      activations = new float[netsize];
      biases = new float[netsize];
      externalCurrents = new float[netsize];
      invTimeConstants = new float[netsize];
      potentials = new float[netsize];
      weights = new float[netsize * netsize];
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
  };
  CTRNN::CTRNN(int netsize, float stepsize) : pimpl{std::make_unique<impl>()}
  {
    pimpl->init(netsize);
    return;
  }
  CTRNN::~CTRNN()
  {
    return;
  }
  float CTRNN::getActivation(int index)
  {
    return pimpl->activations[index];
  }
}
