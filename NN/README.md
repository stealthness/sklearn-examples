# Neural Network Examples

This folder contains examples of using neural networks in sklearn library

## ATT (aka ORL) Example

Using the ATT (aka ORL) Data set. This data set contains 40 subjects, each with 10 images.
Each image is ?? by ?? and is grayscale.

## Iris Example

Using the Iris Dataset which contains 

## Digits Exampl

Using the a sample of the MINST dataset, which contains 8 by 8 grayscale images of digits.

**Conclusions**

The solver options of _lbfgs_ and _sgd_ proved to poor, with accuracy of 0.3417 and 0.2056 respectively.

The solver option of _adam_ proved to be the most successful, and could be improved when _max_iter_ was increased to 1000 from the default 
of 200. It had accuracy of 0.6028.