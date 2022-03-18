import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error


def custom_loss(y_pred, y_true, b1, b0, t, lam):
    err = np.power((y_pred - y_true), 2).mean() + lam * np.linalg.norm(np.array([b0, b1]))
    print("loss", err)
    return t if err >= t else err


def regression_model(x, y, thresh, alpha=0.01, max_iter=1000000, lam=0.5):
    b = np.random.random(2)
    for i in range(max_iter):
        y_pred = b[0] + b[1] * x
        loss = custom_loss(y_pred=y_pred, y_true=y, t=thresh, lam=lam, b0=b[0], b1=b[1])

        # if loss >= thresh:
        #
        #     g_b0 = 1
        #     g_b1 = 1
        # else:
        g_b0 = -2 * (y - y_pred).mean() + 2 * lam * b[0]
        g_b1 = -2 * (x * (y - y_pred)).mean() + 2 * lam * b[1]

        print(f"({i}) beta: {b}, gradient: {g_b0} {g_b1}, g_size:{np.linalg.norm(g_b0 - g_b1)}")

        b_prev = np.copy(b)

        b[0] = b[0] - alpha * g_b0
        b[1] = b[1] - alpha * g_b1

        if np.linalg.norm(b - b_prev) < 0.000001:
            st.markdown(fr"{i} iterations")
            break


    return b


def load_dataset():
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target
    df = pd.DataFrame(dict(MedInc=X['MedInc'], Price=cal_housing.target))
    X = df['MedInc']
    st.dataframe(df)
    fig = px.scatter(df, x="MedInc", y="Price")
    st.plotly_chart(fig, use_container_width=True)
    return X, y


def convex_check(X, y, thresh):
    st.subheader("Is the Loss Function Convex?")
    st.markdown(
        r"if y=a")
    loss = []
    for a in np.linspace(-500, 500, 100):
        loss.append(custom_loss(y_pred=a, y_true=y, t=thresh, b1=0, b0=0, lam=0.5))

    l = pd.DataFrame(dict(a=np.linspace(-500, 500, 100), loss=loss))
    st.dataframe(l)
    fig = px.scatter(l, x="a", y="loss")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Is the Loss Function Convex for b0 and b1?")
    loss, b0, b1 = [], [], []

    for _b0 in np.linspace(-100, 100, 50):
        for _b1 in np.linspace(-100, 100, 50):
            b0.append(_b0)
            b1.append(_b1)
            loss.append(custom_loss(y_true=y, y_pred=(_b1 * X - _b0), b1=_b1, b0=_b0, t=thresh, lam=0.5))

    l = pd.DataFrame(dict(b0=b0, b1=b1, loss=loss))
    st.markdown("Loss for different b0 and b1")
    st.dataframe(l)

    sample = l.sample()
    sample_b1 = sample['b1'].values
    sample_b0 = sample['b0'].values

    st.markdown(fr"Loss for randomly selected constant b1 =  {sample_b1}")
    l_b0 = l[l['b1'] == sample_b1[0]]
    st.dataframe(l_b0)
    fig = px.scatter(l_b0, x="b0", y="loss")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(fr"Loss for randomly selected constant b0 =  {sample_b0}")
    l_b1 = l[l['b0'] == sample_b0[0]]
    st.dataframe(l_b1)
    fig = px.scatter(l_b1, x="b1", y="loss")
    st.plotly_chart(fig, use_container_width=True)

    # fig = go.Figure(data=go.Contour(
    #     z=loss,
    #     x=b0,
    #     y=b1
    # ))
    # st.plotly_chart(fig, use_container_width=True)


def make_regression(X, y, thresh, alpha=0.001, lam=0.5):
    st.markdown(fr"alpha: {alpha}")
    st.markdown(fr"lambda: {lam}")
    beta = regression_model(X, y, thresh, alpha=alpha, lam=lam)
    st.markdown(fr"b0, b1: {beta}")
    y_pred = beta[1] * X + beta[0]
    mse = mean_squared_error(y, y_pred)
    st.markdown(fr"mse: {mse}")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers"), secondary_y=False)
    fig.add_trace(go.Scatter(x=X, y=y_pred, mode="lines", name="Error fit"), secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


st.header("You Do #1")
st.subheader("Dataset")
X, y = load_dataset()
st.subheader("Custom Loss Function")
st.latex(r"L(\beta_0, \beta_1, \theta) =")
st.latex(
    r"\sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 } + \lambda (\beta_0^2 + \beta_1^2), (\lambda > 0) \rightarrow if \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 } + \lambda (\beta_0^2 + \beta_1^2), (\lambda > 0) < \theta")
st.latex(
    r"\theta \rightarrow if \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 } + \lambda (\beta_0^2 + \beta_1^2), (\lambda > 0) >= \theta")
# thresh = st.slider("Loss Threshold", min_value=0, max_value=100000, value=80000, step=10000)

convex_check(X, y, thresh=80000)
st.subheader("Make Regression")
thresh = 80000
lam = st.slider("Regularization Multiplier for L2 (lambda)", 0.0, 3., value=0.5, step=0.5)
make_regression(X, y, thresh=thresh, lam=lam, alpha=0.001)
