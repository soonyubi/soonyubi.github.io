---
title: coursera - Neural Networks and Deep Learning - (2) hidden layer
type: blog
prev: /
next: docs/folder/
math: true
---

## Intro

이번에는 하나의 hidden layer 가 있을때 forward propagation/ backward propagation/ loss 를 계산하고
이전에 사용했던 activation function 과는 다른 tanh를 이용하여 2개의 클래스를 가진 planar data를 분류하고자 한다.

neural network를 구성하는 다음 단계를 거치며, 어떻게 공식이 유도가 되었고, 이 공식을 토대로 코드를 어떻게 구성했는지 그리고 구성하면서 들었던 궁금증을 기술하고자 한다.

1. Neural Network 구조 정의 (# of input units, # of hidden units, etc)
2. 모델 파라미터 초기화
3. 반복:
   - forward propagation
   - compute loss
   - backward propagation to get the gradients
   - update parameters (gradient descent)

### Define Neural Network Structure

Neural network를 정의할 때 가장 먼저 해야할 일은 네트워크의 구조를 설계하는 것이다. 즉, 몇개의 뉴런을 사용할지, 어떤 형태의 레이어 구성을 가질지를 결정하는 과정이다.

이번 실습에서는 입력층(input layer), 하나의 은닉층(hidden layer), 출력층(output layer)로 구성된 간단한 구조를 가진다.

- 입력층 (Input layer, \( n_x \)): 주어진 데이터 X 의 feature 개수
- 은닉층 (Hidden layer, \( n_h \)): 뉴런의 개수는 실험적으로 결정하지만, 여기서는 4개로 설정한다. \( n_h = 4 \)
- 출력층 (Output layer, \( n_y = 1 \)): 분류 문제에서 class의 개수와 같다. 이진 분류(binary classification)이므로, \( n_y = 1 \)

### Initialize Model Parameters

Neural Network의 학습을 진행하기전에 weight와 bias를 초기화하는 과정은 매우 중요하다. 그 이유는 초기화하는 방법에 따라 학습속도가 크게 달라질 수 있기 때문이다.
만약 가중치를 0으로 초기화하게 되면 모든 뉴런이 동일하게 학습되어 의미가 없어진다. 따라서, 작은 랜덤값을 사용해서 대칭성을 깬다.
bias는 뉴런이 학습되는 기준을 조정한다. 0으로 해도 무방하다.

코드로 구현하면 다음과 같다.

```python
def initialize_parameters(n_x,n_h,n_y):
    """
    신경망의 가중치(W)와 편향(b)을 초기화하는 함수

    Arguments:
    n_x -- 입력층 뉴런 개수
    n_h -- 은닉층 뉴런 개수
    n_y -- 출력층 뉴런 개수

    Returns:
    parameters -- 초기화된 파라미터 딕셔너리 (W1, b1, W2, b2)
    """
    np.random.seed(42)

    W1 = np.random.randn(n_h, n_x) * 0.01
    W2 = np.random.randn(n_y, n_h) * 0.01

    b1 = np.zeros((n_h,1))
    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters
```

### Forward Propagation

forward propagation은 입력값 X를 네트워크에 통과시키고 \hat{Y} 을 출력하는 과정이다.
다음의 순서로 진행된다.

\[
Z^{[1]} = W^{[1]}X + b^{[1]}
\]
\[
A^{[1]} = \tanh(Z^{[1]})
\]
\[
Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}
\]
\[
A^{[2]} = \sigma(Z^{[2]})
\]
\[
\hat{Y} = A^{[2]}
\]

{{< callout type="info" >}}
은닉층에서 사용하는 활성화함수는 tanh(x)로 다음과 같다.
\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]
이 함수가 가지는 값의 범위는 (-1,1) 이고, 시그모이드 함수보다 출력값의 범위가 넓어 `vanishing gradient problem`을 해결하는데 더 유리하다.
데이터의 중심이 0을 가지게 되므로 시그모이드보다 학습을 더 효율적으로 진행할 수 있다. 그 이유는, 시그모이드 함수는 (0,1)사이의 값으로 항상 양수이다. 그렇기 때문에
역전파를 진행하면서 가중치 업데이트가 편향될 수 있기 때문이다. 하지만 tanh 의 경우 중심값이 0이고 양수,음수 모두 가지기 때문에 평균이 0에 가깝게 유지된다. 따라서 가중치 변화가 더 균형잡히게 된다.

{{< /callout >}}

### Loss Function

이번 모델에서는 binary classification 을 사용하므로, cross entropy loss function을 사용하여 손실함수를 계산한다.
cross entropy loss function 은 다음과 같이 정의한다.

\[
J = -\frac{1}{m} \sum\_{i=1}^{m} \left( y^{(i)} \log(a^{[2](i)}) + (1 - y^{(i)}) \log(1 - a^{[2](i)}) \right)
\]

- m : # of samples
- y(i) : 실제 레이블 (0 또는 1)
- a[2] : 모델의 예측값 (0~1 사이의 확률값)

a[2]가 y에 가까워질수록 J값은 작아지고, 손실값이 낮아진다. (좋은 예측) <br>
a[2]가 y에 멀어질수록 J값은 커지고, 손실값이 커진다. (나쁜예측)

다음은 loss function을 코드로 구현한 것이다.

```python
def compute_loss(A2,Y):
    """
    크로스 엔트로피 손실 함수 (Cost Function) 계산

    Arguments:
    A2 -- 모델의 예측값 (shape: (1, m))
    Y -- 실제 레이블 (shape: (1, m))

    Returns:
    cost -- 손실값 (scalar)
    """

    m = Y.shape[1] # number of samples

    logprobs = np.multiply(Y, np.log(1-A2)) + np.multiply(1-Y, np.log(1-A2)) # element-wise product
    cost =  -np.sum(logprobs)/m

    cost = np.squeeze(cost)
    return cost
```

{{< callout type="info" >}}
손실함수 계산 시 np.sum + np.multiply 조합을 사용하거나 np.dot 을 사용하여 계산할 수 있다.

```
## np.sum + np.multiply 조합
logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
cost = -np.sum(logprobs) / m

## np.dot 사용
cost = -np.dot(Y, np.log(A2).T) - np.dot((1 - Y), np.log(1 - A2).T)
cost = np.squeeze(cost) / m
```

np.multiply() + np.sum() 을 사용하면 결과가 float(단일 값) 으로 나오므로, 코드가 직관적이다.
np.dot() 을 사용하면 2D 행렬 형태로 반환되므로 np.squeeze() 가 필요하다.
연산량이 적은 np.multiply() 방식이 더 직관적이고, 코드 유지보수에도 좋다.

{{< /callout >}}

### Back propagation

역전파는 Chain Rule을 이용하여 손실함수 J에 대한 각 파라미터의 미분을 구하는 과정이다.
이 미분값들은 이후에 Gradient Descent 를 이용해 각 파라미터를 업데이트하는데 사용된다.

#### 수식유도

{{% steps %}}

### Step 1 : 출력층

\[
\begin{aligned}
&\textbf{출력층 미분 유도} \\
J &= -\frac{1}{m} \sum\_{i=1}^{m} \left[ y^{(i)} \log(A^{[2](i)}) + (1 - y^{(i)}) \log(1 - A^{[2](i)}) \right] \\
\frac{\partial J}{\partial A^{[2]}} &= -\frac{Y}{A^{[2]}} + \frac{1 - Y}{1 - A^{[2]}} \\
\frac{\partial A^{[2]}}{\partial Z^{[2]}} &= A^{[2]} (1 - A^{[2]}) \\
\frac{\partial J}{\partial Z^{[2]}} &= \left( -\frac{Y}{A^{[2]}} + \frac{1 - Y}{1 - A^{[2]}} \right) \cdot A^{[2]} (1 - A^{[2]}) \\
&= A^{[2]} - Y \\
dZ^{[2]} &= A^{[2]} - Y \\
\frac{\partial J}{\partial W^{[2]}} &= \frac{\partial J}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial W^{[2]}} = dZ^{[2]} A^{[1]T} \\
dW^{[2]} &= \frac{1}{m} dZ^{[2]} A^{[1]T} \\
\frac{\partial J}{\partial b^{[2]}} &= \frac{\partial J}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial b^{[2]}} = dZ^{[2]} \\
db^{[2]} &= \frac{1}{m} \sum dZ^{[2]} \\ \\
\end{aligned}
\]

### Step 2 : 은닉층

\[
\begin{aligned}
dA^{[1]} &= W^{[2]T} dZ^{[2]} \\
\frac{\partial A^{[1]}}{\partial Z^{[1]}} &= 1 - A^{[1] 2} \\
\frac{\partial J}{\partial Z^{[1]}} &= \frac{\partial J}{\partial A^{[1]}} \cdot \frac{\partial A^{[1]}}{\partial Z^{[1]}} \\
dZ^{[1]} &= dA^{[1]} \cdot (1 - A^{[1] 2}) \\
\frac{\partial J}{\partial W^{[1]}} &= \frac{\partial J}{\partial Z^{[1]}} \cdot \frac{\partial Z^{[1]}}{\partial W^{[1]}} = dZ^{[1]} X^T \\
dW^{[1]} &= \frac{1}{m} dZ^{[1]} X^T \\
\frac{\partial J}{\partial b^{[1]}} &= \frac{\partial J}{\partial Z^{[1]}} \cdot \frac{\partial Z^{[1]}}{\partial b^{[1]}} = dZ^{[1]} \\
db^{[1]} &= \frac{1}{m} \sum dZ^{[1]}
\end{aligned}
\]

### Step 3 : 결론

우리는 다음과 같은 미분값을 얻어낼 수 있다.

\[
\begin{aligned}
dZ^{[2]} &= A^{[2]} - Y \\
dW^{[2]} &= \frac{1}{m} dZ^{[2]} A^{[1]T} \\
db^{[2]} &= \frac{1}{m} \sum dZ^{[2]} \\
dA^{[1]} &= W^{[2]T} dZ^{[2]} \\
dZ^{[1]} &= dA^{[1]} \cdot (1 - A^{[1] 2}) \\
dW^{[1]} &= \frac{1}{m} dZ^{[1]} X^T \\
db^{[1]} &= \frac{1}{m} \sum dZ^{[1]}
\end{aligned}
\]

{{% /steps %}}

다음은 코드로 나타낸 것이다.

```python

def backward_propagation(parameters, cache, X, Y):
    """
    신경망의 역전파 (Backward Propagation) 구현

    Arguments:
    parameters -- 신경망의 가중치 및 편향을 포함하는 딕셔너리
    cache -- 순전파에서 저장한 값들 (Z1, A1, Z2, A2)
    X -- 입력 데이터 (n_x, m)
    Y -- 실제 레이블 (n_y, m)

    Returns:
    grads -- 각 가중치 및 편향에 대한 미분 값 (Gradient) 딕셔너리
    """

    m = X.shape[1]  # 샘플 개수

    # 순전파에서 저장한 값 불러오기
    A1 = cache["A1"]
    A2 = cache["A2"]
    W2 = parameters["W2"]

    # 1️⃣ 출력층의 미분 계산
    dZ2 = A2 - Y  # (n_y, m)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)  # (n_y, n_h)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (n_y, 1)

    # 2️⃣ 은닉층의 미분 계산
    dA1 = np.dot(W2.T, dZ2)  # (n_h, m)
    dZ1 = dA1 * (1 - np.power(A1, 2))  # (n_h, m) → tanh 미분 적용
    dW1 = (1 / m) * np.dot(dZ1, X.T)  # (n_h, n_x)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # (n_h, 1)

    # 결과 저장
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

```

### Update Parameters

이제까지 우리는 순전파(Forward Propagation) 를 수행하여 예측값을 계산하고,
역전파(Backward Propagation) 를 통해 손실 함수의 미분값(Gradient)을 구했다.

이제 마지막 단계로, Gradient Descent (경사 하강법) 을 사용하여 가중치와 편향을 업데이트하는 과정을 정리하자.

Gradient Descent (경사 하강법) 은 모델이 손실 함수의 최솟값을 찾도록 하는 최적화 기법이다.
즉, 손실 함수 J 가 최소가 되는 가중치 W 와 편향 b 를 찾는 과정이다.

가중치 및 편향을 업데이트하는 기본 공식은 다음과 같다.

\[
\theta = \theta - \alpha \frac{\partial J}{\partial \theta}
\]

- \theta : 업데이트할 매개변수 ( W 또는 b )
- \alpha : 학습률 (learning rate) - 업데이트 크기를 조절하는 하이퍼파라미터
- \frac{\partial J}{\partial \theta} : 매개변수에 대한 손실 함수의 미분값 (Gradient)

이 공식을 이용하여 각 층의 가중치와 편향을 업데이트한다.

```python
def update_parameters(parameters, grads, learning_rate):
    """
    가중치 및 편향을 경사 하강법(Gradient Descent)으로 업데이트하는 함수

    Arguments:
    parameters -- 현재 가중치 및 편향 딕셔너리 (W1, b1, W2, b2)
    grads -- 역전파를 통해 계산된 미분값 딕셔너리 (dW1, db1, dW2, db2)
    learning_rate -- 학습률 (alpha)

    Returns:
    parameters -- 업데이트된 가중치 및 편향
    """

    # 딕셔너리 복사 (원본 데이터 손상 방지)
    parameters = copy.deepcopy(parameters)

    # 업데이트 공식 적용
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]

    return parameters

```

### Neural Network 통합 및 최종 모델 구현

이제까지 우리는 순전파(Forward Propagation), 손실 함수 계산(Cost Computation), 역전파(Backward Propagation), 경사 하강법(Gradient Descent) 적용 등의 개별 단계를 구현했다.
이제 이 모든 요소를 하나의 신경망 모델에 통합하여 최종적으로 동작하는 모델을 구축해보자.

우리는 아래의 과정들을 하나의 함수 nn_model() 에 통합해야 한다.

1. Neural Network 구조 정의
2. 가중치 초기화
3. Gradient Descent 루프 실행
   - 순전파 실행 (Forward Propagation)
   - 손실 함수 계산 (Compute Cost)
   - 역전파 실행 (Backward Propagation)
   - 가중치 업데이트 (Update Parameters)

```python
import numpy as np

def nn_model(X, Y, n_h, num_iterations=10000, learning_rate=0.01, print_cost=False):
    """
    신경망 모델을 학습하는 함수

    Arguments:
    X -- 입력 데이터 (features, m)
    Y -- 레이블 데이터 (1, m)
    n_h -- 은닉층 크기
    num_iterations -- 경사 하강법 반복 횟수
    learning_rate -- 학습률
    print_cost -- True일 경우 1000회 반복마다 비용 출력

    Returns:
    parameters -- 학습된 가중치 및 편향
    """

    np.random.seed(3)

    # 입력 및 출력 크기 설정
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # 1. 가중치 초기화
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 2. Gradient Descent 루프 실행
    for i in range(num_iterations):

        # 순전파 (Forward Propagation)
        A2, cache = forward_propagation(X, parameters)

        # 비용 (Cost) 계산
        cost = compute_cost(A2, Y)

        # 역전파 (Backward Propagation)
        grads = backward_propagation(parameters, cache, X, Y)

        # 가중치 업데이트 (Update Parameters)
        parameters = update_parameters(parameters, grads, learning_rate)

        # 비용 출력
        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost:.6f}")

    return parameters
```

은닉층 크기 n_h 를 변경하면 어떻게 될까?
아래 실험을 통해 은닉층 크기(노드 수)가 신경망의 성능에 미치는 영향을 살펴보자.

```python
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5]

for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title(f'Hidden Layer of size {n_h}')

    # 모델 학습
    parameters = nn_model(X, Y, n_h=n_h, num_iterations=5000, learning_rate=0.01)

    # 결정 경계 시각화
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)

    # 정확도 평가
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)

    print(f"Accuracy for {n_h} hidden units: {accuracy:.2f}%")

```

- 은닉층이 1~2개일 경우 → 학습이 잘 안 됨 (선형적인 결정 경계)
- 은닉층이 3~5개일 경우 → 학습 성능이 향상됨 (비선형적인 결정 경계)
- 은닉층이 너무 많으면? → 과적합(Overfitting) 위험 존재.
