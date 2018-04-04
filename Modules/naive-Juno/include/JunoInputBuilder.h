/*****************************************************************************/
// Author: Xuefeng Ding <xuefeng.ding.physics@gmail.com>
// Insitute: Gran Sasso Science Institute, L'Aquila, 67100, Italy
// Date: 2018 April 7th
// Version: v1.0
// Description: GooStats, a statistical analysis toolkit that runs on GPU.
//
// All rights reserved. 2018 copyrighted.
/*****************************************************************************/
#ifndef JunoInputBuilder_H
#define JunoInputBuilder_H
#include "SimpleInputBuilder.h"
class JunoInputBuilder : public SimpleInputBuilder {
  public:
    std::vector<std::shared_ptr<DatasetController>> buildDatasetsControllers(ConfigsetManager *configset) final;
};
#endif
