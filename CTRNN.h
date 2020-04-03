#pragma once
#include<memory>

namespace ctrnn
{
  class CTRNN
  {
  public:
    CTRNN(int netsize = 0, double stepsize = 0.01);
    ~CTRNN();
  private:
    class impl;
    std::unique_ptr<impl> pimpl;
  };
}
