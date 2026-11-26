---
title: "Machine Learning in Economics (D200)"
subtitle: "Syllabus (Lent 2027)"
author: "Dr. Stefan Bucher"
format: 
  pdf: 
    cite-method: natbib
  html: default
bibliography: auxiliary/references.bib
---

**University of Cambridge** **Faculty of Economics**

**Course Code and Title:** Machine Learning in Economics (D200)\
**Term:** Lent Term 2027\
**Lecturer:** Dr. Stefan Bucher \
**Office Hours:** Wed 3.00pm-4.00pm. Sign up [here](https://calendly.com/stefabu/office-hours)

**Lectures:** Wed 11.00am-1.00pm in Meade Room, weeks 1-9  \
**Classes:** Fri 3.00-5.00pm/5.00-7.00pm in Room 7 (Lecture Block), weeks 3, 5, 7, and 9, and possibly week 8.  \
**Teaching Assistant:** Vahan Geghamyan (vg400@cam.ac.uk) 

**Course Website:** <https://github.com/MLecon/ML-in-Economics> \
**Assignment Submission:** [Github Classroom](https://classroom.github.com/classrooms/195107486-machine-learning-in-economics-2026)\
**Readings of further interest:** [Zotero Group Library](https://www.zotero.org/groups/ml_econ)\
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

The lectures and classes feature examples in [Jupyter](https://jupyter.org/) Notebooks for use on [Google Colab](https://colab.google/).

### Package Management

This course uses [`uv`](https://docs.astral.sh/uv/) for Python package management. To get started:

1. Install `uv` (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

## Prerequisites

Linear Algebra, calculus, probability theory and statistics, as well as programming skills (in Python) are required.



# Contents and Schedule

<!--
## Weekly overview

| Week | Topics | Prince (2023) | Murphy (2022) | Goodfellow et al. (2016) | Sutton & Barto (2018) |
|:-----|:-------|:--------------|:--------------|:-------------------------|:----------------------|
| 1 | Fundamentals; linear regression; probability & information theory | Ch. 1, 2, App. C | Ch. 1, 6, 7, 11 | Ch. 1, 2, 3, 5 | — |
| 2 | Classification & discrete choice; loss functions; optimization; PyTorch; model evaluation | Ch. 5, 6, 8 | Ch. 8, 10, 11 | Ch. 5, 8 | — |
| 3 | Trees, forests, boosting (CART, RF, XGBoost); regularization; interpretability (SHAP) | Ch. 9 | Ch. 18 | Ch. 6, 7 | — |
| 4 | Shallow & deep NNs; backprop; CNNs | Ch. 3, 4, 7, 10 | Ch. 13, 14 | Ch. 6, 8, 9 | — |
| 5 | RNNs, LSTMs; transformers; GPT; LLMs; finetuning | Ch. 12 | Ch. 15 | Ch. 10, 12, 15 | — |
| 6 | Unsupervised learning; clustering & dimensionality reduction; VAE; diffusion | Ch. 14, 17, 18 | Ch. 20, 21 | Ch. 13, 14, 20 | — |
| 7 | Bandits; MDPs; value functions; Q-learning; DQN; PPO; RLHF | Ch. 19 | — | — | Ch. 2, 3, 4, 5, 6 |
| 8 | Synthesis: information-theoretic view; KL divergence | — | Ch. 6 | Ch. 3 | — |
| 9 | ML and economics; prediction vs. inference | — | — | — | — |

*Table: Course weeks with main topics and corresponding textbook chapters. Prince = Understanding Deep Learning; Murphy = Probabilistic Machine Learning: An Introduction; Goodfellow et al. = Deep Learning; Sutton & Barto = Reinforcement Learning: An Introduction.*
-->

## Part 1: Supervised Machine Learning

### Fundamentals and Linear Regression - Week 1 (20 January 2027)

-   A brief overview of AI, ML, and Deep Learning [@prince2023, Chapter 1]
-   Linear Regression: Minimizing mean-squared error using matrix notation [@prince2023, Chapter 2]
  -   sklearn
-   Probability and information theory fundamentals [@prince2023, Appendix C]
  -   Recommended viewing: [Bayes theorem](https://www.3blue1brown.com/lessons/bayes-theorem) 
  -   Helpful Refresher: [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)  


### Model Fitting and Optimization - Week 2 (27 January 2027)

-   Classification and discrete choice 
-   Loss functions [@prince2023, Chapter 5]
-   Optimization: Gradient descent, stochastic gradient descent, Adam optimizer [@prince2023, Chapter 6]
-   Introducing PyTorch
-   Model evaluation: bias-variance tradeoff and overfitting, training/test set and cross-validation, double descent [@prince2023, Chapter 8]
-   Economic applications: 
  - Regression: house price prediction, earnings forecasts, etc.
  - Classification: discrete choice, credit/default risk, etc.
-   Recommended viewing: [Gradient descent](https://www.3blue1brown.com/lessons/gradient-descent) 

### Trees, Forests, Boosting, and Regularization - Week 3 (3 February 2027)

-   Regularization: Explicit and implicit regularization, dropout, transfer learning [@prince2023, Chapter 9]
-   Nonparametric methods: CART (Classification and Regression Trees) 
-   Ensemble methods: bagging (random forests), gradient boosting (XGBoost) [@murphy2022, Chapter 18]
-   Model interpretability: Shapley values and SHAP for feature attribution
-   Prediction vs. inference
  - causal ML topics (e.g., Double/Debiased Machine Learning, causal forests) are covered in D300 
-   Introduction to Research Project  


<!--
#### Topics Not Currently Covered

-   Support Vector Machines (SVM)
-   Kernel ridge regression
-   Gaussian Process Models
-->


## Part 2: Deep Learning, Representation Learning, and Generative AI 

### Artificial Neural Networks: Shallow to Deep - Week 4 (10 February 2027)

-   Nonlinear models (e.g., GLM)
-   Shallow neural networks  [@prince2023, Chapter 3]
-   Deep feedforward neural networks (multi-layer perceptrons) [@prince2023, Chapter 4]
-   Backpropagation [@prince2023, Chapter 7]
-   Convolutional neural networks (CNN) [@prince2023, Chapter 10]
-   Recommended viewing: [But what is a Neural Network?](https://www.3blue1brown.com/lessons/neural-networks); [Analyzing our neural network](https://www.3blue1brown.com/lessons/neural-network-analysis); [Backpropagation](https://www.3blue1brown.com/lessons/backpropagation) and [Backpropagation calculus](https://www.3blue1brown.com/lessons/backpropagation-calculus); [But what is a convolution?](https://www.3blue1brown.com/lessons/convolutions)




### Representation Learning & NLP: Transformers and Foundation Models - Week 5 (17 February 2027)

-   Sequence and time series models: Recurrent neural networks (RNN), Hopfield network, LSTM
-   Word embedding (e.g., Word2Vec)
-   Transformers: Self-attention mechanism, encoder-decoder architecture [@prince2023, Chapter 12.1-12.6]
-   Generative Pre-trained Transformers (GPT) [@prince2023, Chapter 12.7-12.10]
-   Large Language Models (LLM) 
-   Post-training: Finetuning and adaptation (e.g., QLoRA)
-   Economic application: text-as-data for empirical microeconomics (cf. D001 course)
-   Recommended viewing: [LLMs explained briefly](https://www.3blue1brown.com/lessons/mini-llm); [Transformers](https://www.3blue1brown.com/lessons/gpt); [Attention in transformers](https://www.3blue1brown.com/lessons/attention); [How might LLMs store facts](https://www.3blue1brown.com/lessons/mlp)
-   Extra: [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) 

<!--
#### Topics Not Currently Covered

-   Graph Neural Networks (GNN)
-->


### Unsupervised Learning & Generative Models - Week 6 (24 February 2027)

-   Unsupervised Learning [@prince2023, Chapter 14]
-   Clustering and dimensionality reduction: K-means, PCA (latent space intuition)
-   Variational Autoencoders (VAE) [@prince2023, Chapter 17]
-   Diffusion Models [@prince2023, Chapter 18]
-   Economic application: marketing
-   Recommended viewing: [But how do AI images and videos actually work?](https://www.youtube.com/watch?v=iv-5mZ_9CPY) 

<!--
#### Topics Not Currently Covered

- Unsupervised Learning: 
  - Gaussian Mixture Models, Expectation Maximization (vs gradient descent)
  - Dimensionality reduction (ICA)
-   Generative Adversarial Networks (GANs) [@prince2023, Chapter 15]
-->


## Part 3: Reinforcement Learning and Alignment

### Reinforcement Learning - Week 7 (3 March 2027)

-   Multi-armed bandits and bandit gradient algorithm as stochastic gradient descent [@sutton2018, Section 2.8]
-   Reinforcement Learning: Markov Decision Processes (MDP), policies, value functions [@prince2023, Chapter 19]
-   Bellman equations
-   Q-Learning and Deep Q-Networks
-   Proximal Policy Optimization (PPO)
-   Economic application: online experimentation, multi-armed bandit testing, ad allocation, and dynamic pricing

<!--
#### Topics Not Currently Covered

-   Hidden Markov Models (HMM)
-   @silver2016
-   @schrittwieser2020
-   Inverse Reinforcement Learning
-->


### Alignment - Week 8 (10 March 2027)

-   Supervised Finetuning (LORA)
-   Reinforcement Learning from Human Feedback (RLHF): Reward modeling (Bradley-Terry model) for PPO, DeepSeek (GRPO), and Direct Preference Optimization (DPO)
-   The Economics of Finetuning (from API system prompt (``wrapper startups'') to LORA to RLHF to training foundation models)
-   Agentic AI 
-   Extra: [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI); [nanochat](https://github.com/karpathy/nanochat)


## Part 4: ML and Economics


### Information-Theoretic Synthesis, ML and Economics - Week 9 (17 March 2027)

-   Review and synthesis: The information-theoretic lens as a unifying principle [@alemi2024]
-   Unified view of supervised learning, unsupervised learning, and representation learning through KL divergence minimization
-   Prediction vs. estimation/inference [@athey2019]
-   Applications of ML in economics

<!--
#### Topics Not Currently Covered

-   Brief remarks on causal inference (separate module)
-   Matrix completion problem: Consumer choice modeling and application to recommender systems
-   Algorithmic auditing and interpretability as behavioral diagnostics: Using SHAP, attention maps, etc. as tools for understanding what ML models learn about human cognition. 
-->



# Classes and Problem Sets 

Classes are meant to discuss problem sets and questions arising from the lectures as well as (towards the end of the term) the research projects. 
Problem sets are to be submitted in groups of up to 2 students (solo or one partner) on Github Classroom.


# Assessment

Assessment in the course is based entirely on the completion of a small-scale research project, which is assessed via 
a written project report of 3 single-spaced pages (approximately 1500 words) due on 14th April 2027, and 
an oral examination which constitutes a brief (5-7 min.) presentation (slides of presentation to be submitted on same day) and Question and Answer session held on either April 22 or 23.  
All elements are essential and constitute 100% weighting.
<!--(15 minutes per student: 5-7min presentation followed by questions)-->
<!--Schedule 30min per student: 15min for my preparation based on the report, 15min for the oral exam incl. presentation. Total effort: 20hrs over 2-3 days. In 2025 I aimed for 20min per report...-->


# Key Dates
20 Jan 2027
: First lecture

3 Feb 2027 (12 noon)
: Problem Set 1 due <!-- distributed 20 January 2027 -->

17 Feb 2027 (12 noon)
: Problem Set 2 due <!-- distributed 6 February 2027 -->

3 Mar 2027 (12 noon)
: Problem Set 3 due <!-- distributed 20 February 2027 -->

17 Mar 2027 (12 noon)
: Problem Set 4 due <!-- distributed 6 March 2027 -->

14 Apr 2027
: Project report due

22-23 Apr 2027
: Oral examination

<!--
27 Jan 2027 (lecture): Project announced
-->



# Policies

Attendance
: Regular attendance at lectures and classes is mandatory.

Academic Misconduct and AI
: All work submitted must strictly adhere to the University's [Plagiarism and Academic Misconduct Policies](https://www.educationalpolicy.admin.cam.ac.uk/plagiarism-and-academic-misconduct), including on the use of [Artificial Intelligence](https://www.educationalpolicy.admin.cam.ac.uk/plagiarism-and-academic-misconduct/artificial-intelligence-ai). Failure to do so may result in serious disciplinary consequences. Further guidance is available in the [University Statement on AI and Assessment](https://blendedlearning.cam.ac.uk/artificial-intelligence-and-education/generative-ai-and-assessment) and the [AI Policy Framework](https://blendedlearning.cam.ac.uk/ai-policy-framework). 

Late Submissions
: Assignments submitted after the deadline will be penalized (unless an extension is granted in advance) by 10% per 24 hours (additively, i.e. a submission received 49 hours after the deadline will receive 70% of full marks). 

## **Support**

Students are encouraged to use office hours to discuss any academic or personal issues related to the course. Additional support services are available through the university's counseling and academic support centers.

## **Feedback**

Feedback of any kind is most welcome. To suggest improvements (e.g. typos) on the teaching material, please open a Github issue.

# Resources and Reading Materials