#include <iostream>
#include <math.h>
#include "CTRNN.h"

namespace ctrnn
{
  struct CTRNN::impl
  {
    double *activations, *biases, *externalCurrents, *potentials;
    double *invTimeConstants;
    double *weights; // One segment of 'from'-weights per 'to' neuron
    int netsize;
    double stepsize;

    double sigmoid(double in)
    {
      return 1 / (1 + exp(-in));
    }
    
    void init(int netsize, double stepsize)
    {
      this->netsize = netsize;
      this->stepsize = stepsize;
      activations = new double[netsize];
      biases = new double[netsize];
      externalCurrents = new double[netsize];
      invTimeConstants = new double[netsize];
      potentials = new double[netsize];
      weights = new double[netsize * netsize];
      // Init properties of each neuron
      for (int i = 0; i < netsize; i++)
	{
	  biases[i] = 0.0f;
	  externalCurrents[i] = 0.0f;
	  invTimeConstants[i] = 1.0f;
	  potentials[i] = 0.0f;
	  activations[i] = sigmoid(potentials[i] + biases[i]);
	}
      // Init weights (netsize * netsize)
      int from, to;
      for (to = 0; to < netsize; to++)
	{
	  for (from = 0; from < netsize; from++)
	    {
	      weights[netsize * to + from] = 0.0f;
	    }
	}
    }

    void updatePotentialsEuler()
    {
      for (int to = 0; to < netsize; to++)
	{
	  double input = externalCurrents[to];
	  for (int from = 0; from < netsize; from++)
	    {
	      input += weights[netsize * to + from] * activations[from];
	    }
	  potentials[to] += stepsize * invTimeConstants[to] * (input - potentials[to]);
	}
      for (int i = 0; i < netsize; i++)
	{
	  activations[i] = sigmoid(potentials[i] + biases[i]);
	}
    }
    
    void updatePotentialsRK4()
    {
      int from, to;
      double input;
      double *k1 = new double[netsize];
      double *k2 = new double[netsize];
      double *k3 = new double[netsize];
      double *k4 = new double[netsize];
      double *tmpAct = new double[netsize];
      double *tmpPot = new double[netsize];
      // Step 1
      for (to = 0; to < netsize; to++)
	{
	  input = externalCurrents[to];
	  for (from = 0; from < netsize; from++)
	    {
	      input += weights[netsize * to + from] * activations[from];
	    }
	  k1[to] = stepsize * invTimeConstants[to] * (input - potentials[to]);
	  tmpPot[to] = potentials[to] + (0.5 * k1[to]);
	  tmpAct[to] = sigmoid(tmpPot[to] + biases[to]);
	}
      // Step 2
      for (to = 0; to < netsize; to++)
	{
	  input = externalCurrents[to];
	  for (from = 0; from < netsize; from++)
	    {
	      input += weights[netsize * to + from] * tmpAct[from];
	    }
	  k2[to] = stepsize * invTimeConstants[to] * (input - tmpPot[to]);
	  tmpPot[to] = potentials[to] + (0.5 * k2[to]);
	}
      for (to = 0; to < netsize; to++)
	{
	  tmpAct[to] = sigmoid(tmpPot[to] + biases[to]);
	}
      // Step 3
      for (to = 0; to < netsize; to++)
	{
	  input = externalCurrents[to];
	  for (from = 0; from < netsize; from++)
	    {
	      input += weights[netsize * to + from] * tmpAct[from];
	    }
	  k3[to] = stepsize * invTimeConstants[to] * (input - tmpPot[to]);

	  tmpPot[to] = potentials[to] + k3[to];
	}
      for (to = 0; to < netsize; to++)
	{
	  tmpAct[to] = sigmoid(tmpPot[to] + biases[to]);
	}
      // Step 4
      for (to = 0; to < netsize; to++)
	{
	  input = externalCurrents[to];
	  for (from = 0; from < netsize; from++)
	    {
	      input += weights[netsize * to + from] * tmpAct[from];
	    }
	  k4[to] = stepsize * invTimeConstants[to] * (input - tmpPot[to]);
	  potentials[to] += (k1[to] + (2 * k2[to]) + (2 * k3[to]) + k4[to]) / 6;
	  activations[to] = sigmoid(potentials[to] + biases[to]);
	}
    }
  };
  CTRNN::CTRNN(int netsize, double stepsize) : pimpl{std::make_unique<impl>()}
  {
    pimpl->init(netsize, stepsize);
    return;
  }
  CTRNN::~CTRNN()
  {
    return;
  }
  double CTRNN::getActivation(int index)
  {
    return pimpl->activations[index];
  }
  double CTRNN::getBias(int index)
  {
    return pimpl->biases[index];
  }
  double CTRNN::getExternalCurrent(int index)
  {
    return pimpl->externalCurrents[index];
  }
  double CTRNN::getTimeConstant(int index)
  {
    return 1 / pimpl->invTimeConstants[index];
  }
  double CTRNN::getWeight(int fromIndex, int toIndex)
  {
    return pimpl->weights[pimpl->netsize * toIndex + fromIndex];
  }
  void CTRNN::setBias(int index, double bias)
  {
    pimpl->biases[index] = bias;
  }
  void CTRNN::setExternalCurrent(int index, double externalCurrent)
  {
    pimpl->externalCurrents[index] = externalCurrent;
  }
  void CTRNN::setTimeConstant(int index, double timeConstant)
  {
    pimpl->invTimeConstants[index] = 1 / timeConstant;
  }
  void CTRNN::setWeight(int fromIndex, int toIndex, double weight)
  {
    pimpl->weights[pimpl->netsize * toIndex + fromIndex] = weight;
  }
  void CTRNN::updatePotentials()
  {
    //pimpl->updatePotentialsEuler();
    pimpl->updatePotentialsRK4();
  }
}
