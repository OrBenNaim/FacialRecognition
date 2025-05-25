***Facial Recognition Using One shot learning***

***The purpose of the exercise***

- The main goal of this exercise is for you to be able to read, understand, (loosely) implement a scientific paper, recreate its results, and write reports about your experiments.

***Introduction***

- In this exercise, you will use convolutional neural networks (CNNs) to carry out the task of facial recognition - CNNs is the current state-of-the-art approach for analyzing image-based datasets. More specifically, you will implement a one-shot classification solution. Wikipedia defines one-shot learning as follows:
“… an object categorization problem, found mostly in computer vision. Whereas most machine learning-based object categorization algorithms require training on hundreds or thousands of samples/images and very large datasets, one-shot learning aims to learn information about object categories from one, or only a few, training samples/images.”
Your work will be based on the paper Siamese Neural Networks for One-shot Image Recognition. Your goal, like that of the paper, is to successfully execute a one-shot learning task for previously unseen objects. Given two facial images of previously unseen persons, your architecture will have to successfully determine whether they are the same person. While we encourage you to use the architecture described in this paper as a starting point, you are more than welcome to explore other possibilities.

***Instructions***
1. Read the above-mentioned paper.

2. Use the following dataset - [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/index.html)

   - **a.** Download the dataset. 
        Note: there are several versions of this dataset, use the version found [here](https://talhassner.github.io/home/projects/lfwa/index.html) (it’s called LFW-a, and is also used in the DeepFace paper). 

   - **b.** Use the following train and test sets to train your model: [Train](https://drive.google.com/file/d/1Ie-8ihDHfS_FmxAq4EMtMZ-PpMdezQB8/view?usp=drive_link) \ [Test](https://drive.google.com/file/d/11r_8bbGap1skrEzrQtQOu8ZztG6UVisX/view?usp=drive_link). 
   [Remember—you will use your test set to perform one-shot learning. 
   This division is set up so that no subject from the test set is included in the train set]. 
   Please note it is often recommended to use a validation set when training your model. Make your own decision whether to use one and what percentage of (training) samples to allocate.

   - **c.** In your report, include an analysis of the dataset (size, number of examples – in total and per class – for the train and test sets, etc.). 
    Also provide the full experimental setup you used – batch sizes, the various parameters of your architecture, stopping criteria and any other relevant information. 
    A good rule of thumb: if asked to recreate your work, a person should be able to do so based on the information you provide in your report.

3. Implement a Siamese network architecture while using the above-mentioned paper as a reference.

    - **a.** Provide a complete description of your architecture: number of layers, dimensions, filters, etc. Make sure to mention parameters such as learning rates, optimization and regularization methods, and the use (if exists) of batch norm.

    - **b.** Explain the reasoning behind the choices made in answer to the previous section. If your choices were the result of trial and error, please state the fact and describe the changes made throughout your experiments. 
      Choosing certain parameter combinations because they appeared in a previously published paper, is a perfectly valid reason. 

4. In addition to the details requested above, your report needs to include an analysis of your architecture’s performance. 
Please include the following information:
  - **a.** Convergence times, final loss and accuracy on the test set and holdout set
  - **b.** Graphs describing the loss on the training set throughout the training process
  - **c.** Performance when experimenting with the various parameters
  - **d.** Please include examples of accurate and misclassifications and try to determine why your model was not successful
  - **e.** Any other information you consider relevant or found useful while training the model

- Please note that the report needs to reflect your decision-making process throughout the assignment. Please include all relevant information.
