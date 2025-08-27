# Score-Based Generative Modeling through Stochastic Differential Equations (with BASIS separation)

This repo is a fork of the score_sde_pytorch repo (described below) with the addition of source separation. Source separation can be thought of in terms of the "cocktail party problem" in which individual voices are separated from a mix of conversations. In this case, we can attempt to separate individual images from a mash-up of two or more.  To do this, we take advantage of the nature of score-based models, and the BASIS framework introduced in https://github.com/jthickstun/basis-separation.  In general, if we know some forward process to produce **y** from **x**, then score-based models can let us produce **x** from **y**. Here, our mixture is **y** and our set of individual images is **x**.  In this toy implementation, I've used the framework from BASIS separation [Source Separation with Deep Generative Priors](https://arxiv.org/abs/2002.07942) to model a mixture as the sum of two (or more) images.  Then, using a gaussian likelihood and the score-based model as a prior, we can sample separated images from the posterior.  For details, check out the BASIS separation paper and the appendices of [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS). This repo implements BASIS separation on state of the art score-based models, and interfaces with the predictor-corrector framework of the score_sde repo, allowing for tests of different sampling algorithms.  Check out the `Source_Separation.ipynb` notebook for an example using CIFAR-10 images.

![ex1](assets/ex1.jpg)

![ex2](assets/ex1.jpg)

![ex3](assets/ex1.jpg)
