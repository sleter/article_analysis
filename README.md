# System for analyzing web articles using deep neural networks
## Abstract
The work presented in this thesis focuses on accomplishing two things: building a data processing
pipeline and, within this pipeline, solving one particular ML task. The investigated
task is the classification problem of predicting articles’ popularity based on data collected
during the project realization period.
The pipeline implemented in this thesis consists of multiple modules. Data collection
system consists of data harvesting module that uses News API for searching and retrieving
news articles from eleven popular publishers; data enrichment module that uses Facebook
Graph API; preprocessing module that uses various preprocessing tools that help to cope
with many obstacles like categorical attributes, feature scaling or incomplete data, and machine
learning samples maker module. Raw harvested and preprocessed data are analyzed
and visualized as they contain much interesting information about the news articles and
their popularity.
Machine learning samples are used for training, optimizing, and evaluating multiple
deep learning models. Features in ML (Machine Learning) samples are calculated by using
feature engineering and word embedding methods. The approach to solve the articles’
popularity problem includes implementations for dealing with imbalanced dataset and handling
DL (Deep Learning) models overfitting.
Despite the fact that neural networks are trained with data indicating user engagement,
i.e., reaction count in social media, models predict the popularity of a news article using
only its title and author. Such a tool may be useful for publishers that would like to assess
whether the new articles will appeal to readers. The test scenario allows to check the system
as a whole but also to evaluate the operation of individual modules.

Detailed info: [here]()
