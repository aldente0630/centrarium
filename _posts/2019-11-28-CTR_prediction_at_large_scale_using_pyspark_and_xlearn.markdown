---
layout: post
title: CTR Prediction at Large Scale using 'PySpark' and 'xLearn'
date: 2019-11-28 00:00:00
author: Jonas Kim
categories: Data-Science
---  

Suppose you are tasked with doing CTR prediction for a large dataset. It's a dataset that's too large to load in memory on one server. Fortunately, Apache Spark parallel clusters are available, so I've established the following development principles:
* Data preprocessing will proceed using Spark clusters as much as possible.
* An out-of-core learning approach will be performed for model training.
* Prediction work will also be done using Spark clusters.

Here we use the FM / FFM algorithms instead of other classifiers for CTR prediction. This is because they are robust on input sources and provides an absolute predicted CTR value more accurately than the relative rank for classification. The fastest FM / FFM library implemented in the Python language is now [xLearn](https://github.com/aksnzhy/xlearn) so I'll use it. Although not common, I'll start by drawing a model serving pipeline for CTR prediction. 
(To be continued)
