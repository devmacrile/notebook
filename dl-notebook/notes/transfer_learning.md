http://www.di.ens.fr/willow/pdfscurrent/oquab14cvpr.pdf
http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf
http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf
https://arxiv.org/pdf/1503.02531.pdf

Theoretical basis for the earlier activations being more universal??
How do we describe higher level aspects? => in terms of building blocks

Transfer learning makes sense when have a lot of data to learn low level features from, but a small
amount of data for the actual task at hand.
Task A and B have same input X, and there is a lot of data for A, little for B
Low level features from A could be helpful for task B
  -- Yes, but how do you determine if this is possible.. theoretical basis for this?


Multi-task learning
for each class, ask if that object exists in the image
cost function is over each of these 'objects' and their presence
if training over cost function like this, we call it multi-task learning
When?
    Tasks have shared lower level features
    Amount of data for each task is quite similar
Much less common than transfer learning

End-to-End deep learning
    Learning systems that require multiple stages of processing
    Replace that with a single neural network
    e.g.
        audio -> features -> phonemes -> words -> transcript
        to
        audio --------> transcript 



Transfer Learning
ftp://ftp.cs.wisc.edu/machine-learning/shavlik-group/torrey.handbook09.pdf
Too boring

Learning and transferring mid-level image representations...
ImageNet --> PASCAL VOC (i.e. the datasets in the transfer)
"The goal of this work is to show that convolutional network
layers provide generic mid-level image representations that
can be transferred to new tasks."

"the distribution of object orientations and sizes as well as,
for example, their mutual occlusion patterns is very different
between the two tasks. This issue has been also called
“a dataset capture bias”"

Use gridded tiles for pseudo-localization.

tldr; transfer learning works, here's an example.  why? no explanation


Transfer Learning from Deep Features for Remote Sensing and Poverty Mapping
https://arxiv.org/pdf/1510.00098.pdf


How transferable are features in deep neural networks?
http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf
"Many deep neural networks trained on natural images exhibit a curious phenomenon
in common: on the first layer they learn features similar to Gabor filters
and color blobs. Such first-layer features appear not to be specific to a particular
dataset or task, but general in that they are applicable to many datasets and tasks.
Features must eventually transition from general to specific by the last layer of
the network, but this transition has not been studied extensively. In this paper we
experimentally quantify the generality versus specificity of neurons in each layer
of a deep convolutional neural network and report a few surprising results."
Nice, sounds just like what I'm looking for... more empirical than theoretical, maybe, 
but definitely where the heart of the question starts

But...
"Because finding these standard features on the first layer seems to occur regardless of the exact cost
function and natural image dataset"
seems to? I feel like that is the question... why? not 'seems to'

If first-layer features are general and last-layer features are specific, then there must be a transition 
from general to specific somewhere in the network. 
This observation raises a few questions:
• Can we quantify the degree to which a particular layer is general or specific?
• Does the transition occur suddenly at a single layer, or is it spread out over several layers?
• Where does this transition take place: near the first, middle, or last layer of the network?

If the learned features are general, then transfer learning will work ('general' a function of the target).
Okay - and since we've seen this work, we should study the 'nature and extent' of this generality

fine tune vs. frozen
allow errors from new dataset to propagate back into transferred features?  or freeze those features, and just
fit the new fc layers
fine-tuning has potential, but opens the door to overfitting (depending on size of dataset and number of parameters/weights in the source model)
benefits of transfer learning seems roughly independent of number of pre-trained layers and the extent of fine-tuning (this is one of their results)

Generality vs. specificity measured experimentally by splitting the ImageNet dataset (and can do this based on
similarity metrics between the classes).

Split the ImageNet ontology into man-made/natural and can show transfer performance suffers.

Evidence of co-adapting neurons?  (freezing hints at this)
their def: "when neurons on neighboring layers co-adapt during training in such a way that cannot
be rediscovered when one layer is frozen"
could this not be an argument that training was not finished in some way?

a conclusion: "found that even features transferred from distant tasks are better than random weights"

This paper overall is funny to me and in many ways is in-line with many of the "deep learning is epistemologically 
shallow" opinions/arguments.. here we have these models learning from high dimensional data (images) and learning
complex representations, and so the only way we can effectively reason is empirically??




Deep learning at Alibaba
http://static.ijcai.org/proceedings-2017/0002.pdf
Section on transer learning when no assumptions on similarity between source and target can be made
"we propose to learn an explicit transform, from limited data, that explicitly relates instances from the
source domain to those from the target domain"
See [Section 3]
" We then learn a linear transform layer
f(x) = W x, where W is the transformation matrix, that
maps the embedding features x of the source domain into the
target domain. The optimal transform is obtained by minimizing
the distance between images with similar tags and at
the same time maximizing the distance between images with
different tags."
=> incorporating metadata/relationships between actual classes of source and target into the feature reps themselves?
models all the way down... learn the optimal transform using "triplet" loss function (three feature vectors)
but, how to define these triplets?  is this more about distributions/scales being different than overall domain/classes??
This process is O(n^3) (read: infeasible), so have to select hard triplets (based on different classes that are similar)


A theoretical framework for deep transfer learning
https://academic.oup.com/imaiai/article/5/2/159/2363463/A-theoretical-framework-for-deep-transfer-learning



What is the Best Multi-Stage Architecture for Object Recognition?
http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
not transfer learning per se, but random weights and pooling lead to results comparable to actual fit?
not really sure yet


Panel discussion on the math of deep learning
https://cbmm.mit.edu/sites/default/files/documents/Poggio_AAAI17_SoI.pdf
see: https://en.wikipedia.org/wiki/Degeneracy_(mathematics)
