#pragma once
#include<memory>

namespace ctrnn
{
  class CTRNN
  {
  public:
    CTRNN(int netsize = 0, double stepsize = 0.01);
    ~CTRNN();
    double getActivation(int index);
    double getBias(int index);
    double getExternalCurrent(int index);
    double getTimeConstant(int index);
    double getWeight(int fromIndex, int toIndex);
    void setBias(int index, double bias);
    void setExternalCurrent(int index, double externalCurrent);
    void setTimeConstant(int index, double timeConstant);
    void setWeight(int from, int to, double weight);
    void updatePotentials();
  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
  };
}
