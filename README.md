
 ![alt tag](https://cdn-images-1.medium.com/max/1280/1*2ZnWDYFDhoM3QStBoghWeA.png)
 
 # How to convert a trained model into a Core ML Model? 
* Converting Trained Models to Core ML : http://apple.co/2sjpAXw
* Creating an IOS app with Core ML from scratch!by Gerardo Lopez Falcón : http://bit.ly/2hBqU3Y
* Understanding conversion process Udemy by Mohammad Azam : http://bit.ly/2zJUZD3


* How to convert a Scikit learn model : https://youtu.be/T4t73CXB7CU
```python
coreml_model = coremltools.converters.sklearn.convert(model, 'message', 'label')
coreml_model.save('MessageClassifier.mlmodel')

```

* How to convert a Caffe model : https://www.appcoda.com/core-ml-tools-conversion/
```python 
#deploy.prototxt – describes the structure of the neural network.
#oxford102.caffemodel – the trained data model in Caffe format.
#class_labels.txt – contains a list of all the flowers that the model is able to recognize.

coreml_model = coremltools.converters.caffe.convert(('oxford102.caffemodel','deploy.prototxt'), 
image_input_names='data', class_labels='class_labels.txt')
coreml_model.save(coreml_model.save('Flowers.mlmodel'))

```

* How to convert a Keras model : https://github.com/r4ghu/iOS-CoreML-MNIST

```python
# http://bit.ly/2mvq7Dk

import coremltools

output_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
scale = 1/255.
coreml_model = coremltools.converters.keras.convert('./mnistCNN.h5',
                                                   input_names='image',
                                                   image_input_names='image',
                                                   output_names='output',
                                                   class_labels=output_labels,
                                                   image_scale=scale)

coreml_model.author = 'Sri Raghu Malireddi'
coreml_model.license = 'MIT'
coreml_model.short_description = 'Model to classify hand written digit'

coreml_model.input_description['image'] = 'Grayscale image of hand written digit'
coreml_model.output_description['output'] = 'Predicted digit'

coreml_model.save('mnistCNN.mlmodel')

```



 
 
