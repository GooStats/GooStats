/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef JunoDatasetControllers_H
#define JunoDatasetControllers_H
#include "SimpleDatasetController.h"
class JunoDatasetController : public SimpleDatasetController {
  public:
    JunoDatasetController(ConfigsetManager *_c) : SimpleDatasetController(_c) { };
    //! we need to set fraction and coefficient here
    bool collectInputs(DatasetManager *) final;
};
#endif
