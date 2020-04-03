#include <iostream>
#include <math.h>
#include "CTRNN.h"

namespace ctrnn
{
  struct CTRNN::impl
  {
    float *activations, *biases, *externalCurrents, *invTimeConstants, *potentials;
    float *weights;
    int netsize;
    float stepsize;

    float sigmoid(float in)
    {
      return 1 / (1 + exp(- in));
    }

    float getWeight(int fromIndex, int toIndex)
    {
      return weights[toIndex * netsize + fromIndex];
    }
    
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
	  biases[i] = 0.0f;
	  externalCurrents[i] = 0.0f;
	  invTimeConstants[i] = 0.0f;
	  potentials[i] = 0.0f;
	  activations[i] = sigmoid(potentials[i] + biases[i]);
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
  float CTRNN::getBias(int index)
  {
    return pimpl->biases[index];
  }
  float CTRNN::getExternalCurrent(int index)
  {
    return pimpl->externalCurrents[index];
  }
  float CTRNN::getTimeConstant(int index)
  {
    return 1 / pimpl->invTimeConstants[index];
  }
  float CTRNN::getWeight(int fromIndex, int toIndex)
  {
    return pimpl->getWeight(fromIndex, toIndex);
  }
}
