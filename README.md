# 개 고양이 이미지 분류
## 인공지능

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

32x32x1 흑백으로 구성된 강아지, 개  이미지 10000개를 가지고 인공신경망을 학습시켜 분류하는 프로젝트 입니다.


- 수업시간에 배운 이론을 활용하여 강아지와 고양이를 분류하는것이 목표.
- 4명 구성원 (컴퓨터과학과) 1개월 작업.

## Tech 
- Keras
- loss function / optimizer / batch size / epochs / learning rate 
- Dense / Conv2D / MaxPooling2D / Flatten / Dropout / Activation




 
## Review
- 첫 Keras Project로 모델을 구성하는 방법에 대해 배웠다.
  기본적으로 네트워크 구성함에 있어 Desne, Convolution MaxPooling를 가지고 조합을 시도하였고 여기에 더해 loss function, optimizer, batch size, epochs, learning rate 등이 사용되는것을 알 수 있었다. 
- 먼저 Convolution 네트워크과 MaxPooling 을 고정을 잡고, 네트워크 외의 변수를 iterative 방식으로 접근하여 validation_accuracy가 가장 높은 최적의 값을 찾아냈다.
- 그리고 네트워크를 구성함에 있어서 Convolution 네트워크와 MaxPooling의 조합 또한 iterative 방식으로 접근하여 validation_accuracy가 가장 높은 최적의 값을 찾아냈다. 여기에서 순서를 어떻게 두느냐, 몇개의 조합을 두느냐에 따라 결과가 많이 차이 났다. 그 결과에 맞추어 Convolution, Max Pooling을 조합하고 마지막으로 dropout Layer를 추가했다.
-  프로젝트에서는 BatchNormalization을 사용 금지하였다. 그런데 과제로 나온 이 문제와 같은 간단한 Model에서 학습시 Gradient Vanishing / Exploding의 문제가 나타날 수 있어 Internal Covariance Shift 현상이 일어날 수 있는데 이를 해결하기 위해서 Whitening 의 문제를 해결해야 했고 결국 BatchNormalization의 문제를 해결하기 위해서 dropout Layer를 여러 Layer로 두어 유사 BatchNormaliztion을 구현하는 방법으로 정확도를 많이 끌어 올렸다. 하지만 Overfitting에 대한 문제는 결국 해결하지 못하였다. 단순 dropout layer를 중복 추가하는것만으로는 Overfitting의 문제를 피하지 못하였다.
- 처음 써보는 Keras 프로젝트였고  모델을 어떻게 구성하느냐에 따라 결과 값이 많이 달라졌다. iterative 접근이라고 했지만 사실상 여러번의 for문을 돌면서 optimize하는 작업이 였기 때문에 이거를 자동화 할 수는 없나는 생각이 들었고 github에 이미 Keras의 매개변수 나아가 Layer까지 Optimize하는 Method를 제공하고 있는 Repository를 보았다. Hyperparameter tuning 뿐만 아니라 layer까지 알아서 최적으로 구성하는점에 있어서 네트워크 구성까지 Supervised Learning으로 구성할 수 있지 않을까 하는 생각을 했다.


## Result
![DeployDiagram](/result.png)

## License

MIT



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
