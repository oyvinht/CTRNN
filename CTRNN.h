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
  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };
}
