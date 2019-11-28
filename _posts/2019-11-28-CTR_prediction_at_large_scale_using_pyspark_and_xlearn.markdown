---
layout: post
title: CTR Prediction at Large Scale using 'PySpark' and 'xLearn'
date: 2019-11-28 00:00:00
author: aldente0630
categories: Data-Science
---  

Suppose we need to perform CTR prediction for a large dataset. It's a dataset that's too large to load in memory on one server. Fortunately, Apache Spark parallel clusters are available, so we can establish the following development principles:
* Data preprocessing will proceed using Spark clusters as much as possible.
* Out-of-Core Learning approach will be taken for model training.
* Prediction work will be also performed using Spark clusters.

We use the FM / FFM algorithms instead of other classifiers for CTR prediction. This is because they are robust on input sources and provides an absolute predicted CTR value more accurately than the relative rank for classification. The fastest FM / FFM library implemented in the Python language is now [xLearn](https://github.com/aksnzhy/xlearn) so we'll use it. Although not common, let's start by drawing a model serving pipeline for CTR prediction.
