#pragma once
#include<memory>
#include<math.h>

float sigmoid(float in)
{
  return 1 / (1 + exp(-in));
}

namespace ctrnn
{
  class CTRNN
  {
  public:
    CTRNN(int netsize = 0, float stepsize = 0.01);
    ~CTRNN();
    float getActivation(int index);
    float getBias(int index);
    float getExternalCurrent(int index);
    float getTimeConstant(int index);
    float getWeight(int fromIndex, int toIndex);
    void setBias(int index, float bias);
    void setExternalCurrent(int index, float externalCurrent);
    void setTimeConstant(int index, float timeConstant);
    void setWeight(int from, int to, float weight);
    void updatePotentials();
  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };
}
