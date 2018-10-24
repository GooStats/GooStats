/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef GPUManager_H
#define GPUManager_H
#include "Module.h"
class GPUManager : public Module {
  public:
    GPUManager() : Module("GPUManager") { }
    bool preinit() final;
  private:
    bool report(bool siliently = false) const;
    bool report(int gpu_id,bool siliently = false) const;
};
#endif
