#Stack Overflow Closed Questions

The idea around this project is simple. Get a dataset of questions asked on Stack Overflow and determine based on title and questions and other features if the question is closed or not.

This actually comes from a Kaggle competition but it is still a good NLP exercise. In the previous projects we used extremely simple algorithms such as Naive Bayes, Logistic Regression, and Random Forests (With word Embeddings thanks to a not so simple algorithm Word2Vec) and achieved great results.

This project is actually going to step up the tools. We will now be diving into the world of deep learning. There are two paths we could go down, Tensorflow/Keras combo or PyTorch. I personally think the syntax, while less pythonic, of Tensorflow is easier to read. With that in mind, this project will use Tensorflow. In the future, might come back and rework it for PyTorch but if nothing else, a future project will use PyTorch.

The setup is easy, go get dataset. Determine which features to use and then feed the data into the model and see how close we get. I have seen several benchmarks for this task at about 60-70% accurate so this is our target range.
