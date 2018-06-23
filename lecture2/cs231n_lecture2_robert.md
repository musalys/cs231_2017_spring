---
layout: post
title: 'CS231n 2강 내용 정리'
author: robert
date: 2018-06-23 14:38
category : playwings
tags: [deeplearning, cnn, CNN ,neural networks, convolutional neural networks]
published: false
---

CS231n Lecture 2 | image classification
====



## Image Classification

 ![what](../images/lecture2_title.png "blahblah")

#### Image Classification: a core task in computer vision

컴퓨터 비젼에서의 핵심 태스크

컴퓨터는 이미지를 받아들일 때

1. 고양이 이미지를 input으로 투입
2. 시스템이 이미지에 카테고리와 레이블을 알고있다고 가정하고, 그런 카테고리의 고정된 데이터 셋이 있음
3. 따라서 해당 이미지를 보고 고양이라는 cat label을 부여

#### The Problem: Semantic Gap

직면하는 문제 : Semantic Gap (의미 차이)

컴퓨터는 이미지를 보고 전체적으로 고양이라고 해석하지 않고, 컴퓨터는 거대한 숫자의 grid를 본다.

e.g 800 * 600 * 3 (RGB)

여기서 Semantic Gap 발생하는데, 'cat'이라는 의미는 우리가 부여한 것이다.
따라서 이러한 의미에 의해 부여한 '고양이'라는 label과 숫자로 이루어진 image의 value들 사이에는 큰 Gap이 존재한다.


#### Challenges: Viewpoint variation (앵글 변화)

이것은 큰 문제가 되는데 카메라 앵글이 바뀌게되면, 위의 Grid의 값들은 변경이 되지만, 여전히 의미에서 '고양이'이다.


#### Challenges: Illumination (조명 변화)

또 다른 한 가지의 문제는 조명이다. 빛의 유무(명암)에 따라서, 조금씩 값이 바뀌지만 여전히 고양이 이다.


#### Challenges: Deformation (포지션 변화)

또 다른 문제는 포지션(고양이의 포즈)

#### Challenges: Occlusion (폐색)

또 발생할 수 있는 문제는 폐색인데 우리는 고양이의 매우 일부만 볼 수도 있다.

#### Background: Clutter (배경과 유사)

우리가 분류하려는 대상이 배경과 유사할 수 있다.

#### Intraclass variation (클래스 내부 변화)

클래스 내부에서도 변화가 있다. 같은 고양이라도 종이 다른 고양이라던지

무늬가 다른 고양이가 생기는 문제
따라서 우리의 알고리즘은 위의 이러한 변화에 대해서 강건(robust)해야 한다.

#### Slide image classifier

파이썬으로 이미지 분류기 API를 작성할 때

대충 슬라이드와 같은 프로그램을 짜겠지만

중간에 알고리즘은 모호하다. 직관적이다 따라서 작성하기 어렵다.

#### Slide Attempts have been made

이미지를 보고 가장자리 (윤곽선)을 알아내는 방법이 있었는데,
이것은 아래와 같은 문제가 있었다.
1. 매우 취약하다
2. 다른 object들에 대해서 하면 처음부터 다시 시작해야한다. (확장성 X)

#### Slide Data-Driven Approach

1. 이미지와 레이블에 대한 데이터 셋 수집
2. 머신 러닝을 사용해 분류기를 학습
3. 새로운 이미지를 기존의 분류기 투입해 평가

 이미지 분류기 API 만드는 새로운 접근 방법으로 데이터가 사용

 두가지 함수로 제작

 1. train
 2. predict


#### Slide First classifier: Nearest Neighbor

1. train -> Memorize all data and labels
2. predict -> Predict the label of the most similar training image

#### Slide Example Dataset: CIFAR-10

10 classes
50K training images
10K testing images

#### Distance Metric to comapare images

L1 distance: d1(I1,I2) = Sigma(abs(I1-I2))

pixel-wise absolute value differences and sum all (manhattan distance)

#### Code: Nearest Neighbor classifier

N sample이 있을 때

Train은 단순 입력으로 O(1)
predict는 비교 이므로 O(N)

Train time을 길게가져가고, Predict time을 짧게 가져가는 것이 목적

#### Slide K-Nearest Neighbors

가장 가까운 이웃의 label을 복제하는 대신, k 최인접 point들의 majority vote를 통해 label을 결정

K가 증가할 때마다 decision boundary들이 smooth 해짐

white space는 클래스를 판단 할 수 없는 영역

#### Slide K-Nearest Neighbors: Distance Metric

L2 (Euclidean) Distance

#### Slide Hyperparameters

가장 적절한 K 값은 얼마인가?
가장 적절한 distance metric은?

매우 problem-dependent 하다
다 해보고 무엇이 베스트인지를 알아내는 방법밖엔 없다

#### Setting Hyperparameters

idea 1: 데이터에 가장 잘 work하는 hyperparameter를 고른다. k=1은 항상 best fit이므로 BAD
idea 2: train / test 데이터 셋으로 나누고 test data에 best hyperparameter를 고른다.
idea 3: train / validation / test 셋으로 나누고 val / evaluate test에서 best hyperparameter를 고름 better
idea 4: Cross-Validation: fold로 데이터를 나누고 각각을 평가해 평균내서 사용
> 작은 데이터셋에서는 사용되나, 딥러닝에서는 잘 사용되지 않음

#### Slide k-NN on images never used.

- 테스트에 매우 느림
- 픽셀의 Distance metrics가 정보를 주지 않음

맨 왼 사진과 오른쪽 3사진과 똑같은 L2 distance value를 가짐

- Curse of dimensionality

차원이 높아질 수록 지수승으로 data point가 늘어나기 때문에
high dimensional space를 다 밀집하게 채울 만한 이미지를 가지기 어렵다 -> 연산 측면에서 아주 불리해진다.


#### Slide K-Nearest Neighbors: Summary

이미지와 레이블의 트레이닝 셋으로 이미지 분류를 하려면, 테스트 셋에 대해 레이블을 예측해야한다.
KNN 분류기는 가장 가까운 트레이닝 example을 통해 레이블을 예측한다

Distance metric과 K는 hyperparameters이다.

validation set에서 사용한 hyperparameter를 고르고 test는 마지막에 딱 1번만 수행해라

### Linear Classification

#### Slide Parametric Approach: Linear Classifier

Image -> f(x,W) -> 10 numbers giving class scores

전체 데이터를 입력해서 학습시킬 필요 없이
점수 계산만 하면 됨 -> small device에 적합

f(x,W) = Wx + b

#### Slide Interpreting a Linear Classifier

f(x,W) = Wx + b

아래 그림은 CIFAR-10에서 학습한 weight들의 이미지들인데
하나의 클래스에 대한 어떤 템플릿을 만든다고 보면 됌(해당 클래스 이미지들의 특징들을 모아놓음)

나중에 딥러닝으로 가면 single template이 아니라 여러 teamplate을 사용할 수 있음

이것은 high dimensional space를 linear하게 나누는 작업이다.

#### Slide Hard cases for linear classifier

1. 두개 클래스가 만드는 decision boundary는 첫번째 그림과 같이 나누어지는데, 이는 하나의 선형 boundary로 표현할 수 없다.
2. 비선형의 영역일 경우 문제가 된다 클래스의 영역이 비선형이라면 이는 선형 boundary로 표현을 할 수 없다.
3. 세 고립된 영역일 경우도 역시 하나의 선형 boundary로는 표현을 할 수 없다.

#### So far: Defined a (linear) score function f(x,W) = Wx + b

우리가 여태까지 한 것은 각 Class에 대한 점수를 계산했다
그러나 이 점수가 좋은지 나쁜지는 어떻게 구분하나? 다음시간에 알아볼 것이다.
