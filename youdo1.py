import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import mean_squared_error


def custom_loss(y_pred, y_true, t):
    err = np.power((y_pred - y_true), 2).mean()
    return t if err >= t else err


def regression_model(x, y, thresh, alpha=0.01, max_iter=100000):
    b = np.random.random(2)
    for i in range(max_iter):
        y_pred = b[0] + b[1] * x
        if custom_loss(y_pred, y, thresh) >= thresh:
            g_b0 = 0
            g_b1 = 0
        else:
            g_b0 = -2 * (y - y_pred).mean()
            g_b1 = -2 * (x * (y - y_pred)).mean()

        print(f"({i}) beta: {b}, gradient: {g_b0} {g_b1}, g_size:{np.linalg.norm(g_b0 - g_b1)}")

        b_prev = np.copy(b)
        b[0] = b[0] - alpha * g_b0
        b[1] = b[1] - alpha * g_b1

        if np.linalg.norm(b - b_prev) < 0.000001:
            print(f"Stopped at iteration {i}")
            break

    return b


st.subheader("Dataset")

cal_housing = fetch_california_housing()

X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target

df = pd.DataFrame(dict(MedInc=X['MedInc'], Price=cal_housing.target))
X = df['MedInc']
st.dataframe(df)
fig = px.scatter(df, x="MedInc", y="Price")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Loss Mean Squared Error")
st.latex(r"\sum_{i=1}^{N}(x_i-y_i)^2")
st.subheader("Is Loss Mean Squared Error Convex?")
st.latex(r"y=a")

loss = []
thresh = 80000


for a in np.linspace(-100, 100, 100):
    loss.append(custom_loss(a, y, thresh))

l = pd.DataFrame(dict(a=np.linspace(-100, 100, 100), loss=loss))
st.dataframe(l)
fig = px.scatter(l, x="a", y="loss")
st.plotly_chart(fig, use_container_width=True)


loss, b0, b1 = [], [], []
for _b0 in np.linspace(-100, 100, 50):
    for _b1 in np.linspace(-100, 100, 50):
        b0.append(_b0)
        b1.append(_b1)
        loss.append(custom_loss(y, (_b1 * X - _b0), thresh))

l = pd.DataFrame(dict(b0=b0, b1=b1, loss=loss))
st.dataframe(l)

# fig = go.Figure(data=go.Contour(
#     z=loss,
#     x=b0,
#     y=b1
# ))
# st.plotly_chart(fig, use_container_width=True)

sample = l.sample()
sample_b1 = sample['b1'].values
sample_b0 = sample['b0'].values
print(sample)
print(sample_b1)
print(sample_b0)

l_b0 = l[l['b1'] == sample_b1[0]]
st.dataframe(l_b0)
fig = px.scatter(l_b0, x="b0", y="loss")
st.plotly_chart(fig, use_container_width=True)

l_b1 = l[l['b0'] == sample_b0[0]]
st.dataframe(l_b1)
fig = px.scatter(l_b1, x="b1", y="loss")
st.plotly_chart(fig, use_container_width=True)

beta = regression_model(X, y, thresh)
y_pred = beta[1] * X - beta[0]
mse = mean_squared_error(y, y_pred)
print(mse)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=df["MedInc"], y=df["Price"], mode="markers"), secondary_y=False)
fig.add_trace(go.Scatter(x=X, y=y_pred, mode = "lines",name="Error fit"), secondary_y=True)
st.plotly_chart(fig, use_container_width=True)

