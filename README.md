---
title: "Machine Learning in Economics (D200)"
subtitle: "Syllabus (Lent 2025)"
author: "Dr. Stefan Bucher"
format: 
  pdf: 
    cite-method: natbib
  html: default
bibliography: references.bib
---

**University of Cambridge** **Faculty of Economics**

**Course Code and Title:** Machine Learning in Economics (D200)\
**Term:** Lent Term 2025\
**Lecturer:** Dr. Stefan Bucher \
**Office Hours:** Thu 11.30am-12.30pm. Sign up [here](https://calendly.com/stefabu/office-hours)

**Lectures:** Fri 9.00-11.00am in Meade Room, weeks 1-9  \
**Classes:** (some) Thu 4.00-6.00pm in Room 7 (Lecture Block), weeks 3, 5-7, and 9.  \
**Teaching Assistant:** Cheuk Ng (cfn24@cam.ac.uk)

**Course Website:** <https://github.com/MLecon/ML-in-Economics> \
**Assignment Submission:** [Github Classroom](https://classroom.github.com/classrooms/195107486-machine-learning-in-economics-2025)\
**Readings:** [Zotero Group Library](https://www.zotero.org/groups/ml_econ)\
**Recordings of further interest:** [Youtube Playlist](https://youtube.com/playlist?list=PLo8op7DIq2yhVzg8sUVAc36PRDZVOw6GD&si=ZV1YOyqJCbzlFdF9)

# Course Overview

## Course Description

Machine Learning is in the process of transforming economics as well as the business world. This course aims to provide a graduate-level introduction to machine learning equipping students with a solid and rigorous foundation necessary to understand the key techniques of this fast-evolving field and apply them to economic problems. The curriculum bridges theoretical foundations with practical implementations and strives to remain relevant by teaching a conceptual understanding that transcends the implementation details of current state-of-the art methods.

## Specific Topics Covered

The course covers key methods in

-   supervised learning, including regression, classification, and neural networks
-   unsupervised learning, including clustering and dimensionality reduction
-   reinforcement learning, including bandit problems
-   applications to economics.

## Course Aims and Objectives

By the end of this course, students will be equipped with:

-   a foundational understanding of the most relevant ML tools and how they are reshaping economic analysis
-   the ability to work with ML models using popular software environments such as [PyTorch](https://pytorch.org) and [scikit-learn](https://scikit-learn.org), and to adapt them for economic problems
-   critical skills in interpreting and explaining sophisticated ML models in economic contexts

## Lecture Materials

Lecture materials will be posted to the course website. 
The course loosely follows the textbook of @prince2023 which is freely available at <https://udlbook.github.io/udlbook/>. The material may be complemented by chapters from further classic textbooks, including @bishop2006, @hastie2009, @goodfellow2016, @mackay2003, @murphy2022, and @sutton2018. These are not required reading.

All readings are also organized in a [Zotero Group Library](https://www.zotero.org/groups/ml_econ). A [Youtube Playlist](https://youtube.com/playlist?list=PLo8op7DIq2yhVzg8sUVAc36PRDZVOw6GD&si=ZV1YOyqJCbzlFdF9) curates videos of further interest to the course's topics.

## Computing Environment

The lectures and classes feature examples in [Jupyter](https://jupyter.org/) Notebooks for use on [Google Colab](https://colab.google/). Students with a demonstrated need can request HPC access (e.g. for the project) after consulting with the instructor.

## Prerequisites

Linear Algebra, calculus, probability theory and statistics, as well as programming skills (ideally in Python) are required.



# Contents and Schedule

### Introduction and Foundations - Week 1 (24 January)

-   A brief overview of AI, ML, and Deep Learning [@prince2023, Chapter 1]
-   Probability and information theory fundamentals [@prince2023, Appendix C]


## Part 1: Supervised Machine Learning

### Prediction and Linear Regression - Week 2 (31 January)

-   Linear Regression: Minimizing mean-squared error using matrix notation [@prince2023, Chapter 2]
-   Optimization and stochastic gradient descent [@prince2023, Chapter 6]
-   Model Evaluation: Bias-variance tradeoff and overfitting, training/test set and cross-validation, double descent [@prince2023, Chapter 8]
-   Regularization: Ridge and Lasso regression [@prince2023, Chapter 9]
-   sklearn

### Classification and Logistic Regression - Week 3 (7 February)

-   Multi-Layer Perceptron
-   Multinomial Logit and Discrete Choice [@prince2023, Chapter 5]
-   Support Vector Machines (SVM)
-   Decision/classification trees
-   Ensemble Methods: Boosting and bagging, random forests, gradient boosting machines

### Artificial Neural Networks and Deep Learning - Week 4 (14 February)

-   Deep learning as a nonlinear model (like GLM) [@prince2023, Chapter 3]
-   Feedforward neural networks (=multi-layer perceptrons) [@prince2023, Chapter 4]
-   Backprop and stochastic gradient descent [@prince2023, Chapter 7]
-   Convolutional neural networks (CNN) [@prince2023, Chapter 10]
-   pytorch

## Part 2: Unsupervised Machine Learning

### Unsupervised Learning - Week 5 (21 February)

-   [@prince2023, Chapter 14]
-   Gaussian Mixture Models, Expectation Minimization (vs gradient descent)
-   Clustering: K-means
-   Dimensionality reduction: PCA and ICA
-   \*Generative Adversarial Networks (GANs) [@prince2023, Chapter 15]




### Generative AI and Natural Language Processing (NLP) - Week 6 (28 February)

-   Sequence and time series modeling: Recurrent neural networks (RNN), Hopfield network, LSTM
-   Word embedding (e.g. Word2Vec)
-   Variational Autoencoders (VAE), variational Bayesian methods, ELBO [@prince2023, Chapter 17]
-   Transformers and Large Language Models (LLM) [@prince2023, Chapter 12]
-   Diffusion Models [@prince2023, Chapter 18]
-   Introduction to Research Project  

## Part 3: Reinforcement Learning

### Reinforcement Learning - Week 7 (7 March)

-   Hidden Markov Models (HMM) and Markov Decision Processes
-   Multi-armed bandit testing
-   Bandit Gradient Algorithm as Stochastic Gradient Descent
-   Q-Learning
-   SOTA RL algorithm (e.g. PPO)
-   Deep Q-Networks
-   @prince2023 [Chapter 19]
-   @silver2016
-   @schrittwieser2020

## Part 4: ML and Economics



### Consumer Choice Modelling - Week 8 (14 March)

-   ML and Economics [@athey2019]
-   Matrix completion problem: Consumer choice modeling and application to recommender systems
-   Inverse Reinforcement Learning

### ML and Economics - Week 9 (21 March)

-   Review and synthesis
-   The information-theoretic lens as a unifying principle [@alemi2024]
-   Brief remarks on causal inference (separate module)
-   Project presentations


# Classes and Problem Sets 

Classes are meant to discuss problem sets and questions arising from the lectures as well as (towards the end of the term) the research projects. 
Problem sets are to be submitted in groups of 4 students (of varying configuration) on Github Classroom.

<!--
6 February: PS 1 (due 4 February)
20 February: PS 2 (due 18 February)
27 February: PS 3 (due 25 February)
6 March: project ideas
20 March: final review and project questions
-->

# Assessment

Assessment in the course is based entirely on the completion of a small-scale research project.

The grade is composed of three submissions, due at 12 noon UK time.

-   10% Proposal (due 10th March 2025): 1-page description of the proposed project, clearly articulating the research questions and the methods to be used.
-   30% Draft (due 24th March 2025): Complete draft of the project report.
-   60% Submission (due 7th April 2025): Final project report incorporating feedback received in response to the draft.

The project report should be around 4 single-spaced pages, approximately 2000-2500 words. 



# Key Dates
24 Jan
: First Lecture

4 Feb 
: Problem Set 1 due <!-- distributed 24 Jan-->

18 Feb   
: Problem Set 2 due <!-- distributed 7 Feb-->

25 Feb  
: Problem Set 3 due <!-- distributed 14 Feb-->

10 Mar 
: Project proposal due <!-- announced 28 Feb-->

21 Mar
: Project presentations (last lecture) 

24 Mar 
: Draft of project report due

7 Apr 
: Final project report due 





# Policies

Attendance
: Regular attendance at lectures and classes is mandatory.

Plagiarism
: All work submitted must be original. Plagiarism will result in serious academic penalties in line with University policy.

Use of Large Language Models
: Submitted work must not be direct output from Large Language Models such as ChatGPT, and may be checked accordingly.

Late Submissions
: Assignments submitted after the deadline will be penalized (unless an extension is granted in advance) by 10% per 24 hours (additively, i.e. a submission received 49 hours after the deadline will receive 70% of full marks). 

## **Support**

Students are encouraged to use office hours to discuss any academic or personal issues related to the course. Additional support services are available through the university's counseling and academic support centers.

## **Feedback**

Feedback of any kind is most welcome. To suggest improvements (e.g. typos) on the teaching material, please open a Github issue.

# Resources and Reading Materials