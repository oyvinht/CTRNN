#include <iostream>
#include <math.h>
#include "CTRNN.h"

__global__
void ctrnn_CUDA_updatePotentialsEuler(int netsize,
				      float stepsize,
				      float *activations,
				      float *biases,
				      float *externalCurrents,
				      float *potentials,
				      float *invTimeConstants,
				      float *weights)
{
  int to = blockIdx.x * blockDim.x + threadIdx.x;
  float input = externalCurrents[to];
  // TODO: Maybe try to put connected neurons into same blocks somehow
  for (int from = 0; from < netsize; from++)
    {
      input += weights[netsize * to + from] * activations[from];
    }
  potentials[to] += stepsize * invTimeConstants[to] * (input - potentials[to]);
  activations[to] = 1 / ( 1 + exp(-(potentials[to] + biases[to])));
}

namespace ctrnn
{
  
  struct CTRNN::impl
  {
    float *activations, *biases, *externalCurrents, *potentials;
    float *invTimeConstants;
    float *weights; // One segment of 'from'-weights per 'to' neuron
    int netsize;
    float stepsize;
    int numBlockThreads; // TODO: Number of cores per SM
    int numBlocks; // TODO: Find number of SM on card (all threads in block run on same SM)

    
  void init(int netsize, float stepsize)
    {
      this->netsize = netsize;
      this->stepsize = stepsize;

      cudaMallocManaged(&activations, netsize * sizeof(float));
      cudaMallocManaged(&biases, netsize * sizeof(float));
      cudaMallocManaged(&externalCurrents, netsize * sizeof(float));
      cudaMallocManaged(&invTimeConstants, netsize * sizeof(float));
      cudaMallocManaged(&potentials, netsize * sizeof(float));
      cudaMallocManaged(&weights, netsize * netsize * sizeof(float));

      // Init properties of each neuron
      for (int i = 0; i < netsize; i++)
	{
	  biases[i] = 0.0f;
	  externalCurrents[i] = 0.0f;
	  invTimeConstants[i] = 1.0f;
	  potentials[i] = 0.0f;
	  activations[i] = 0.0f;//sigmoid(potentials[i] + biases[i]);
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
      int maxBlocks = 4;
      numBlockThreads = 32 * 4;
      numBlocks = ceil(netsize / (float) numBlockThreads);
      std::cout << "Initialized net of size " << netsize << std::endl;
      std::cout << "Processing config:" << std::endl;
      std::cout << "  Parallel blocks:         \t"  << numBlocks << std::endl;
      std::cout << "  Threads per block:       \t" << numBlockThreads << std::endl;
    }

   void updatePotentialsEulerCUDA()
    {
      ctrnn_CUDA_updatePotentialsEuler<<<numBlocks, numBlockThreads>>>
	(netsize,
	 stepsize,
	 activations,
	 biases,
	 externalCurrents,
	 potentials,
	 invTimeConstants,
	 weights);
    }
    
    void updatePotentialsRK4()
    {
      int from, to;
      float input;
      float *k1 = new float[netsize];
      float *k2 = new float[netsize];
      float *k3 = new float[netsize];
      float *k4 = new float[netsize];
      float *tmpAct = new float[netsize];
      float *tmpPot = new float[netsize];
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
  CTRNN::CTRNN(int netsize, float stepsize) : pimpl{std::make_unique<impl>()}
  {
    pimpl->init(netsize, stepsize);
    return;
  }
  CTRNN::~CTRNN()
  {
    return;
  }
  float CTRNN::getActivation(int index)
  {
    cudaDeviceSynchronize();
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
    return pimpl->weights[pimpl->netsize * toIndex + fromIndex];
  }
  void CTRNN::setBias(int index, float bias)
  {
    pimpl->biases[index] = bias;
  }
  void CTRNN::setExternalCurrent(int index, float externalCurrent)
  {
    pimpl->externalCurrents[index] = externalCurrent;
  }
  void CTRNN::setTimeConstant(int index, float timeConstant)
  {
    pimpl->invTimeConstants[index] = 1 / timeConstant;
  }
  void CTRNN::setWeight(int fromIndex, int toIndex, float weight)
  {
    pimpl->weights[pimpl->netsize * toIndex + fromIndex] = weight;
  }
  void CTRNN::updatePotentials()
  {
    pimpl->updatePotentialsEulerCUDA();
    //pimpl->updatePotentialsRK4();
  }
}
