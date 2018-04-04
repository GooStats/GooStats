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
class GPUManager {
  public:
    static GPUManager *get();
    bool report(bool siliently = false) const;
  private:
    bool report(int gpu_id,bool siliently = false) const;
    static GPUManager *fGPUManager;
};
#endif
