# Design of the software

The core business of the software is to compose the [SumLikelihoodPdf](../PDFs/include/SumLikelihoodPdf.h) object and
feed it to the fitting engine [GooFit](https://github.com/GooStats/GooFit).

The non-trivial part of the software is to be able to perform global-fit with shared parameter among different dataset
at arbitrary granularity.

In order to achieve this functionality, the two fundamental Entities in GooStats
are [Configsetmanager](../Kernel/include/ConfigsetManager.h) and [DatasetManager](../Kernel/include/DatasetManager.h)

## Mechanism of fitting parameter sharing

The Tree structure are used to manage the parameter sharing. User is responsible to create the leaf nodes. Parent nodes
will be created automatically based on the `name` parameter of the leaf. For example, if two leaf are named as
`PhaseI.MLPalpha.Top` and `PhaseI.MLPalpha.Bottom`, then two leaf node will be created, sharing the same parent node
named as `PhaseI.MLPalpha` as well as grandparent node `PhaseI`.

The tree is implemented simply using a map.

All nodes, including parent nodes or leaf nodes, are represented by `BasicManager`. Parameters of each leaf will be
stored in the leaf or its parent. If a parameter, say, `lightYield`, is stored in its parent node, say, `node A`, all
descendant nodes of `node A` will share the same `lightYield`. In other words, this parameter is synchronized among all
descendant leaf nodes of this node.

## The ConfigsetManager class

Each `ConfigsetManager` correspond to one configuration file, or set. It inherits
the [OptionManager](../Kernel/include/OptionManger.h) class.

It is at the same time a leaf node in the fitting parameter sharing tree. It inherits
the [ConfigsetManager](../Kernel/include/ConfigsetManager.h)

For example, if you combine different experiment, or dataset with completely different models, you should 
put them in different `ConfigsetManager`. If you combine several data-taking runs with similar levels of backgrounds, you can
either put them in different `ConfigsetManager`. 

Each `ConfigsetManager` can correspond to one or more `DatasetManager`. Usually, if you perform binned fit, one 
`ConfigsetManager` correspond to one piece of Poisson likelihood and a few pull terms. The Poisson likelihood and 
each pull term correspond to one `DatasetManager`, respectively. 

## The DatasetManager class

The `ConfigsetManager` holds the raw information loaded from configuration files.

The `DatasetManager` holds processed information directly needed to construct the piece of likelihood it is 
responsible for.

The [DatasetController](../Kernel/include/DatasetController.h) class is responsible for the process of converting raw 
information to processed information.