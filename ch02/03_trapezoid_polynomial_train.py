# 多项式回归
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def make_data():
    np.random.seed(10)
    x1 = np.random.randint(5, 10, 50).reshape(50, 1)
    x2 = np.random.randint(10, 16, 50).reshape(50, 1)
    # 得到了一个50行2列的样本数据，其中第1列为上底，第2列为下底。
    x, y = np.hstack((x1, x2)), 0.5 * (x1 + x2) * x1
    return x, y

def train(x, y):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_mul = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_mul, y)
    print("权重为：", model.coef_)
    print("偏置为：", model.intercept_)
    print("上底为{}，下底为{}的梯形的真实面积为{}".format(5, 8, 0.5 * (5 + 8) * 5))
    x_mul = poly.transform([[5, 8]])
    y_pred = model.predict(x_mul)
    print("上底为{}，下底为{}的梯形的预测面积为{}".format(5, 8, y_pred))

if __name__ == '__main__':
    x, y = make_data()
    train(x, y)