#pragma once
#include<memory>

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
  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };
}
