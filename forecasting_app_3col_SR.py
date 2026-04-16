# Time Series Forecasting App
# Author: Sierra Reynolds | Utah State University
# Track 1 – Core App

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError as e:
    st.error(f"Missing dependency: {e} — run: pip install -r requirements.txt")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════
import gzip as _gzip, base64 as _b64, io as _io
_CPI_B64  = "H4sIAFM/4WkC/2Wcya4lPW6E936W9kFq4rBstL0w0IsGbL//q5hDBOsHDNSmPlBKxUklFWIq73/8/X/+82//+Nd//f1///Hf//y35Vf//Vvx7297/eQ6yAZRXSCnyP6dDXABzB/IAxFjIwFxvyBa5PzsMcZAnB07QwRkfUXu76qBLBLDiNcGsa86Nqp6MWIFYYz7ATmM6WsZZcnv2xfkgezpWUB8sZWC2HOQlqXRD0DLst9yaQJZ9vsYA1n2ux8GCFn221LSnbLsZ/eBdIz/zrsgB2QbW5Ws8/2+HqBDVhIlEZBr7FlB9CMpWScmxlzLQd6nTVpXEQGpMZ+YPHZANshX5H3QFcT2A2HMNQU5IF63PUnrOj8VgJZ1f/uwGwE5NXuSKIh8F8RAes4lcZK9mkAW50qSBaKPpIcsvwaLquQny0EYcvD/1qQxYEZckDPkgTy5IC0q5wWJgujHaxuI1/1M4owRbQJR/ufqEOU/vei5Rd3v1zdmQ1SAtQWEIa+etSSHpAe4ISuIsZtSdWMuCUmpuvu37wJRkDUXNxAVxjiIGUirSlJJJUkP+WCWJukh39/SuhOHsu7vWySMsZrJSQ76sZZ1KOtFenCQB3L6Thzqer/rjFEQ3yQGIstAWpf8bis91BWJ5ypIjzlmhmOE0BWzpxpdyoqQIyAMqaycoFXZT/oRuVQVWWY9kEeiAC3KQxT7VRAxdlyiXk4dB3GQD920pgDXcaXWFEQNjVpTkQQPml6sWP10PmgK4vuAlKgXU6ev/SAqyMX/H/5/JkJAXAFK0jtxLwXEQK7x0g7ihhhoiuVJ0DE0xSzZjNkgXxOhqEwwB4QxTxXkMOawVYuK5Wkx5oG8x5iWFVOgf1GhLv2tyuxJjDHCVg6i/bMLdQWp1TtJjzkyysPVoSuyzisVtBcvckyroL0I8hlAyZKYFcaQi5D3GPMQc4WkZMmKJLhAFDGdoukuIqRvH91FgKcYXosK0ovco7sowpgasOxIH9Ux3UWQqw+kY04kLwc5IKtTKd2FRNJxA3mI6dlEcxEhW3gpBfHFSxmIDGlZOXdAIEvgLh7dRcRwNFAVU8fr4jQXQaQnJc2FhI9xkoMY61+H5qJiAB6APzZqVQYT9egtgjz83/B/uxvEh2gTaIqUc3BpaIrVqvMxnUWSujFCZ6EfFnehswjSs1boLIIMKE264lIG8kB6tgmdhUaG+dixgvQ6KHQWmkuagJQsPfFr3SYtSw+WPaGzCCKKflqWhospzym0Fppp54AwZl2SA3JrogjNRRAlaFmRmSppC71FEN3spmXFXFoCYiDX2cpBnqAVZMVKVI+50Fsos47QW6iFrGpFcxExXo+I0FwEscpMQnMRrVbLorkIYs5+HkjvPYTuQh1ZUOgugtgmMRBXXr10GZ2z0F0EWbU5ErqLIPZhPK3L0gNXP3QXlksWyZ+YA1K6LCfLBrkg7QqE7iLIewAlyyLL6AVRkNtDprlIshZIy7p/WkHW/b2LS0FWLFG1tAjNheWeqvqhu4hW3tLpLjKGoFXF7Ok7SncRBMOhuwiite0S2guL3dFhq5YV86nnAe1FEKuUK7QX0WqVvxD6iyBnIwaygpRLF/oLS6dcrWgwLHMPQIfEJmuRHJDX3dBfeEyVlk6H4bH89O2jw8iY2rEILUbErMdWhpheeIUWwxcSn9BieEwVjqdlJcFwWlWA2noIDYbnZNogHRL2pjMGDUaQfUguiBG0qPDJtWwI/YXnsvZAWlRMgk579BeePpnE0Uo3+oGo8MA9U+gvPLdYaAVVucWq8dBfBDn91NBfOLfjQoPhMXc6wdNgON2N0GAEuf1c02B4zMFaMoUGwzNfPRADecqY1hU2eGGE0BVTZeHq0BXTiSE15PWFVS7vKbQYibwHRI9R6DHqEL3azQpdxvpiSm1GPSLvh4dGI5EYkTYKR3zZlxHhDtBrJHr989JsrC+SUNWHhG4joy6DNoPawYmPxNxzbaA9qAfvIzHnzgW6jHrOvt6fKCIhEmf3kBizZRoao7CS+kiMXXlnKB+JygzgIzFmjBBh9IaJr99oDI+8iCbKK93pNxod80i/0RjLU22ZlR5krazRMKo1rsgxNXGULiSRVS5Q2pBEtx4bpQ9ZK/fNQNC40tFgqNC40q5gENC4DjbuSi+yVmSAepaVZiTRrk2v0o0kMmVUa1xx084FekTyiKAxHvqJUiI7HIQRec0vXaMxVghHQ2qMJYKjp0b7I4gaHbtH3aMxkBBNVCcspTVZOx7Hw6hL5D0naE7it0I2VLqTRNI/NO3J2vkIbSAjOrxgS9y5/jeBwiCnzJDSoCSyw6ge+45NRPVEh7J2rPdluJUWJRFGcEZgLOfGhpdo1b5QzwiMBd3ZlxD5ZUMIzI0po4zIetqcUWhhF6QRJcYOsh5GPSPRYRiUbmWd71ebZb0jMR6pqnoq7UoGdSlNaVjWyYflAV0iLXurtCyR72I+LCAhkkWk/Jm1MqPekRhoBuH8nY9hEJT4sBzpHYmxYWND3sVA5R70jUbBXlDf3EaB0dU3t1FRztQ3tzGQET0i6dn15jZalwP1jUSD9dc3Eu03Q5ib6J3f3l9u4nV0/ucmdr1F3yiMZ6wqWkofk7cM803+chd7S6zyl7v4GdGfu9gzXOYuRq5URsmg/h1oZ/LG7p5c9DOFHhs6kfUPQUezzqRPWppEzzF6TtSDFVV1NEb6LKOoOhovNjmqozGm0iKCxhf5gH09ojdIBvWs19EYD3bfNB2NgjVPdTRK3NndiBrjWf8QRY2xfj6MnhrDcPWzMQbnWNcqdfxNkC6r6PibE2m3s+f4m4h6vXKNv4mo05lk/M1x7FF1/M2N5bMsgo6/Oc61cvxN1bcxCEjMcjZGCoX3w25Ex9/cdPLV+/ibQLfqHjr+JlCXC3X8zd1wpzr+5qahAnlD+vaMvQn0lcXXsTc3/Tt7N6K2fjr2JpAe9EWFfDmmY29uzKSFwVPiRd3axt5kkbwWEht7c2M9KNU29iYL3vX429ibiGpzZmNvsmE9Uzb2JlC/s7CxN1dRibGxN1nkrh/Vxt5c7UqHjbu5Md2MaBEtyKHC8AJVkbMxNzdmoG6gTdSvkGzMzc3SgQBdIq1V18bcZNQmaoUv8xT7UqJVt9/G3CT6OIhWmBVuDgISswxu6B4SH0sINuYmUJtnG3Pz8gWJA22iqpTaeJusfVehysbbvDR+II/k1NyysTYvpwiREvU7XRtrkzXyvofjbQJ1zrAxNy/TGxpSYRjGymU25iYb1rDG3GRVvIPG3GQRfKIOUe9JbcxNIKkKh425eYbV2sbcZENnlDLqY1dGYn0rxttkcVwwBiqMvIWeKDDNKIN67MKylI23SVS1DxtvE6jfVNt4G4l5VFbaxttkkbyfuvE2gfrljo23kTwlwKhWKLkE8opG1MupjbcJhB9wvI0cWBQbb1PIgDaR9JQfbyOcbmNtkvTgx9pUeZxRl6hdmI21kYeEZ2NtJGeNAimRdYYYbyMSOV2AnKjfR9mYm0DOwVNibhcxVEpUvGa2MTeB2qPYmBsxGHMbcxOob8Z4myCrtk023kYMLzJsvI04XpbaeBv98H7GxtsE+mpRtPE2WSE3IEjMwvrCICBRI//UEmjjbTTLDDWI8TaJuvvxNhWlQIfIOr2Nt9GYSgLyhnROH2sTSDu7jbVR1rlsrE2gt9kXJMa8qQ2ejbXJcjmGQIVBDO2oMLxUORQbZxOoXx/YWJtE3ftYm6qiK9Alev2bjrUJdB6REHn/gGNtNA+csHsjWkYEiYZKnY21ydJ5T5LxNhm1iTD62F2VJ7LxNllh7wVivI06c/94G81qmABdIquNoI25sTBmnZXG3CQydt8ajXUtG3MTCHd2zE3W1WsHaWNuAmlnhDE3licLcEVoDGTlnXzMjYXDKo0+5iaivMqNPuYmol6tST7mxm4s6uzrMUqmL2FUJQQfb5OkVg0fb2O5LrKdE/VhIh9zU0iBFlHvy3zcjQkOw/i4myy9V97wcTeJFqMgUWGmfdxNoFV7Qx93E8g/NhQiU0ZBYxbgGWVEnZZ83E0W5TcQNcb0Yl/Q6B8K/j7uJqK8cqOPu4moPprm4248twIXqDX6+u3Fhpeoj5X5+Bv/+vSEj73xhRKpj73xjVMCPvbGWTX1sTd+UF/2sTeJBGOgxJhdRrSJ+rCUj7/x+3uHaA+q7OXjbzzfJS+gO6geFx9/k7V6EiHpc0w+9sbDqxuIkexL5ETKIVBh+KlKLj7+JtAMlAoD1esyH3/jhoKij7/xTFREhw37/IGPv6kS/QN6RDINITEmVz9A42/csTo7/c3+PlQjnP4mUVf8nP6mogyjx9G+LxJV5UGnv8moLgw7/U1GdZnJaXD2l6UHBTqMapPlNDgZ1fsrp8HZX84uRglRl+mdBieR9rR8o/HCKvsbjbeLPv5G4oUf8DcSL1Z6fyPxot7mMhKzJi9Am6jfhLqMRMFbcpeRKHg76jISBYcGXEaiYtPqMhJjxvWjISNRsbi4jETFPtZlNCoKyi6jMXwXr0iNsTR2jpgDtYFuz945Ufs5b+McqV089eJzqDZQWwmfU7WJlKg1rphL070yqo+F+hyszRcDi907EQkkrg1P6nO2duUhBqJNJI3mdG2+F+hZP8drI6oMtc/x2nVRP/I5XxvtvKfgHLDNNwXTlRBpL8VzxDZQ7699ztjmy4PFvnwQCBU+prw5ZrvCrpef9jlnu7JeXWOYg7Yr7ZMCTRTmyBy1jajvMOoSyce+HpEroyDRuNL7SAyfr+weEh2lIfeRmO8hbiNqzNwlQIvo1StG99EYVixSY+Qd+pu9MwWB7CGRBhudQR+jcPh71VnJRo9IBuH4dx+HaaREud9tZIOmeyfKA5yFIDFfc4TERotRaozajDKrK66RmKdiiDaRhjtvBI2vsn+jS3T9AL2Jmu5loq4DQaPUgZVGRnTDbTZyojxmU4gatUx2o0WUNqXRJpJVaI9GrS1PI0TlGRkHOkRZJGt0ifQSQWOmM/YlRE8USIm0Z86eg/xfbbsbOaPyYGkhaDxZ2sLooTFft+gB2uxLa/D0NxmUe8ZGm8i3AJ0/yIBaYh4jPyCPRAa1wpPH8C6QDjJeEArzlSYv6ET3oiEVXs7nMwpvrXmNMPhXJ58C3ZGY9dYNhKg8xfeAzqDDqEt0JuoR+WX30JjlLgFSot3PxhicY3XWq5EzSh6iqDHr9A60iNQwCGoMV9/P/xicUwcfmmySxZhWmPX384DuoB7V2JsswE/nrTAL8GsBKaOsn4yxNxH1AThBfmRRiF+cpOkCWQzKDN4In5OcX92HsTZZxxeiibF+msba3NwHCtCdKCeCvNhA6gUSIulnbqxNFt8frwh5r466N/JBG31R4KvDlY0w+pxrGAQV9kGtQGNtbr53P0AT9S7IIfH+IcbZ5KHzjwgSs6zFroSoE+AYmzybvhkEhV7nrBr5IMNAqTBmGhtSoTM7/PlsKOthhf58N/TVO89Gm2g7Eb4cynfEDnSJcp/R6BFZD2KsTZ06v0BKZH1jx9q8/tSlkRPdi+6hMaM6A463yRr9IdpE3jlqvE2W3z+iTXQm6hD1AjjWJoksIEjMRKZAMsgZpUTWy+RYm5eJjN07UZ5cLESJ+VIaZJF8bQ7G2bxIRxm0xtg8q1fEjSboKNH5/wgK/S8NH1FPrjXO5uX7nwOET8Bi2swgjCidWiMnqiS5xthku3pe1xibrNFvXBAKJdNPNRxjk0fZxYEQteHB1hibPLrubHgH1axZY2zyqHqlyTXGJqvv05cS1Yq4xtcEuTMsKIzEdVcjSozZxt4p8dar8kYYfJ+BCDS+RvotdaOJuvVErfE1eY79gVwSqTVyja3JejybQWDMLHcgJZKabWtcTVboBzmRO3qnwqzjI4oK81AREcbep1YCja0JdMqArbE1gVbfsbE14vVxRqOWmPX48ndrfI32ScBGrVEXnrE1vkbzLTWIkVxhUEvUHTcRvUNilvY3GkJiRN3yZGtsTZ6AL6u4xtboRopdY2vyxPtxoEOktdSssTVKU7bG1mSJ/rJ7ITotaGxNIGM7SHyxmrKdE/lDO0p8dcCiEQafGQkNKVF4F8fVaB44PUCI0jpl3OgQYeqOr9E85byAHqOsH+vxNZpTCQQKvb4RbGQMssdhOaPWxRj4zelX37c2WkSnn+FxNnXovVSPt8lyfLnaNd7G8jOKB3QG9bwZb5NFe5JHcpW9C5EJe2+JtusUWyMjUmNDfFV76vOxQpR4YCvWWJtAE7SH9LMyzsbyyxsB2kTqRIeof5lxNnng/bCrR/Q6U46zsX7d2AgKs6D1gKAwDPnH7p1Rz9EXFWYuQxQVRsPNqE30+vaMtakD8xcIUVaHdxpBolddqtFllJV7W2NtMqqMxhprY/ntKIj+IQxqiVl4X4xyotfrwzibQMfREBI9d4AYFiR6pKl+9sfZZNRHtInciA4bElwCJPAxNnUUnkiI8sOmRkqEZXKMjefBCQdyIlP0RYW3TqM0WkTed3+cTR6Ar1Vkj7VxFiH2WBt/9fVjIygUeI891iaPxR/2BY1aHzE0Ekb5ZZQyaiu7h8b07gLkRLU+7LE2QXyhd0rMDyvQjhLzy9JCY23cI0NcoP5sOs/H18+158vp70PG2/Pp9LfqqGgjfOeeH9WACEm75j0fT39xFy9Rfz2d59DL9e35fPq7dRq2EL6fzkPnl6g/oM46NceAz8IT9c2gtTl5wrxV75Go9bq+ESRafZPfCBLzYwI2fER2GdUa86B45alNc5NIypXvPV/zp6t0IHzOv+tkUyF+z59LM7qHxkCCkULiykJIIXqbWJV/q/a+m97mrJyVG+gQndZDb3OyploL+Ka3SWT1TO0zEq1OjDRSIjtsCIler/4atcT9VeW1ECQmqtyyz/zNggWJZ/5mwUbG23f+aEFMpdr87Dt/tWBh/d53/mwBC1Wb3qZQz3B6m+q+suCmt8moo2yoHKk7r2iM6jVw379I3D36O3fR62P2RotReQyn0Z6oWMD/D+QdBwUsRAAA"
_MORT_B64 = "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"
def load_demo(name):
    raw = _gzip.decompress(_b64.b64decode(_CPI_B64 if name == "cpi" else _MORT_B64)).decode()
    return pd.read_csv(_io.StringIO(raw))

def compute_metrics(actual, predicted):
    actual    = np.array(actual,    dtype=float)
    predicted = np.array(predicted, dtype=float)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return {"MAE": round(mae,4), "RMSE": round(rmse,4), "MAPE (%)": round(mape,4)}

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def make_lag_features(series, n_lags=6):
    df = pd.DataFrame({"y": series.values})
    for lag in range(1, n_lags+1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["rolling_mean_6"] = df["y"].shift(1).rolling(6).mean()
    df.dropna(inplace=True)
    return df

# ══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════════════
def run_holt_winters(train, test, fh):
    m = ExponentialSmoothing(
        train, trend="add", seasonal="add",
        seasonal_periods=12, initialization_method="estimated"
    ).fit(optimized=True)
    tp = m.forecast(len(test))
    tp.index = test.index
    future_idx = pd.date_range(test.index[-1] + pd.DateOffset(months=1), periods=fh, freq="MS")
    fp = pd.Series(m.forecast(len(test)+fh).values[-fh:], index=future_idx)
    return tp, fp

def run_arima(train, test, fh):
    m = ARIMA(train, order=(1,1,1)).fit()
    tp = m.forecast(steps=len(test))
    tp.index = test.index
    future_idx = pd.date_range(test.index[-1] + pd.DateOffset(months=1), periods=fh, freq="MS")
    fp = pd.Series(m.forecast(steps=len(test)+fh).values[-fh:], index=future_idx)
    return tp, fp

def run_random_forest(train, test, fh, n_lags=6):
    full    = pd.concat([train, test])
    feat_df = make_lag_features(full, n_lags)
    n_tr    = len(feat_df) - len(test)
    X_tr, y_tr = feat_df.iloc[:n_tr].drop("y",axis=1), feat_df.iloc[:n_tr]["y"]
    X_te       = feat_df.iloc[n_tr:].drop("y",axis=1)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_tr, y_tr)
    tp = pd.Series(rf.predict(X_te), index=test.index)
    history = list(full.values)
    future_preds = []
    for _ in range(fh):
        row = {f"lag_{l}": history[-l] for l in range(1, n_lags+1)}
        row["rolling_mean_3"] = np.mean(history[-3:])
        row["rolling_mean_6"] = np.mean(history[-6:])
        pred = rf.predict(pd.DataFrame([row]))[0]
        future_preds.append(pred)
        history.append(pred)
    future_idx = pd.date_range(test.index[-1] + pd.DateOffset(months=1), periods=fh, freq="MS")
    fp = pd.Series(future_preds, index=future_idx)
    return tp, fp

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════════
def make_forecast_plot(train, test, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Train",
                             line=dict(color="#4B9BFF", width=2)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual (Test)",
                             line=dict(color="#00CC88", width=2)))
    colors = ["#FF6B6B", "#FFB800", "#A78BFA"]
    for i, (name, (tp, _)) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(x=tp.index, y=tp.values, name=f"{name} (Test)",
                                 line=dict(color=colors[i], width=2, dash="dash")))
    for i, (name, (_, fp)) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(x=fp.index, y=fp.values, name=f"{name} (Forecast)",
                                 line=dict(color=colors[i], width=2, dash="dot")))
    fig.update_layout(
        title=dict(text="Actual vs Forecast", x=0.5),
        xaxis_title="Date", yaxis_title="Value",
        hovermode="x unified", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Time Series Forecasting Studio",
                       page_icon="📈", layout="wide")

    st.title("📈 Time Series Forecasting Studio")
    st.caption("Utah State University · DATA 5630 Final Project · Sierra Reynolds")

    page = st.sidebar.radio("Navigate", ["🔮 Forecast", "📖 About"])

    if page == "📖 About":
        st.header("📖 About This App")
        st.markdown("""
        A time series forecasting tool built for the DATA 5630 final project at Utah State University.
        Upload any CSV with a date and numeric column to generate econometric and ML forecasts.

        ### 📦 Models
        | Model | Type | Library |
        |---|---|---|
        | Holt-Winters (ETS) | Exponential smoothing with additive trend + seasonality | statsmodels |
        | ARIMA(1,1,1) | Classic differencing model | statsmodels |
        | Random Forest | Ensemble ML with lag + rolling-mean features | scikit-learn |

        ### ✂️ Data Split
        Simple holdout split — you control train % via the slider. The remaining data is the test set.

        ### 📐 How to Read the Metrics
        - **MAE** — Mean Absolute Error. Average error in original units. Lower = better.
        - **RMSE** — Root Mean Squared Error. Penalizes large errors more than MAE. Lower = better.
        - **MAPE** — Mean Absolute Percentage Error. Scale-independent percentage. Lower = better.

        ### 💡 Demo Datasets
        Two built-in datasets are available — no download needed:
        - **US CPI (1947–2024)** — Monthly Consumer Price Index from FRED
        - **30-Year Mortgage Rate (1971–2024)** — Monthly average from FRED
        """)
        return

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 1 — Configuration (full-width horizontal bar)
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("⚙️ Configuration")

    c1, c2, c3, c4, c5, c6 = st.columns([1.4, 1, 1, 1.2, 1.2, 1], gap="medium")

    # ── Data source ───────────────────────────────────────────────────────────
    with c1:
        st.markdown("**📂 Data Source**")
        demo_choice = st.selectbox(
            "Demo dataset",
            ["— Upload my own —", "🏠 30-Yr Mortgage Rate (1971–2024)"],
            help="Pick a built-in demo or upload your own CSV."
        )
        if demo_choice == "— Upload my own —":
            uploaded = st.file_uploader("📁 Upload CSV", type=["csv"], label_visibility="collapsed")
        else:
            uploaded = None

    # Resolve raw dataframe
    raw = None
    if demo_choice != "— Upload my own —":
        key = "mort"
        raw = load_demo(key)
        with c1:
            st.caption(f"✅ Demo loaded · {len(raw):,} rows")
    elif uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            with c1:
                st.caption(f"✅ {len(raw):,} rows loaded")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
    else:
        st.info("👆 Select a built-in demo dataset above — or upload your own CSV — to get started.")
        return

    # ── Column selectors ──────────────────────────────────────────────────────
    with c2:
        date_col     = st.selectbox("📅 Date column", raw.columns.tolist())
        numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found.")
            return
        target_col = st.selectbox("🎯 Target variable", numeric_cols)

    with c3:
        mv         = st.selectbox("🔧 Missing values", ["Forward fill", "Interpolate", "Drop rows"])
        use_filter = st.checkbox("Filter last N periods")
        n_periods  = st.number_input("N periods", 24, 500, 120, disabled=not use_filter)

    with c4:
        train_pct = st.slider("✂️ Training %", 60, 95, 80)
        st.caption(f"Train **{train_pct}%** · Test **{100-train_pct}%**")

    with c5:
        st.markdown("**🤖 Models**")
        run_hw = st.checkbox("Holt-Winters (ETS)", value=True)
        run_ar = st.checkbox("ARIMA(1,1,1)",        value=True)
        run_rf = st.checkbox("Random Forest (ML)",  value=True)

    with c6:
        fh      = st.selectbox("🔮 Horizon (steps)", [12, 24, 36])
        st.write("")
        run_btn = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

    # ── Parse dataframe ───────────────────────────────────────────────────────
    try:
        df = raw.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        df.index = df.index.to_period("M").to_timestamp()
        df = df[[target_col]].dropna()
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 2 — Data preview
    # ══════════════════════════════════════════════════════════════════════════

    if mv == "Forward fill":
        df[target_col] = df[target_col].ffill()
    elif mv == "Interpolate":
        df[target_col] = df[target_col].interpolate("linear")
    else:
        df = df.dropna(subset=[target_col])

    if use_filter:
        df = df.iloc[-int(n_periods):]

    series = df[target_col].astype(float)

    st.subheader("📊 Data Preview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observations", f"{len(series):,}")
    m2.metric("Mean",         f"{series.mean():.2f}")
    m3.metric("Min",          f"{series.min():.2f}")
    m4.metric("Max",          f"{series.max():.2f}")

    fig_prev = px.line(x=series.index, y=series.values,
                       labels={"x": "Date", "y": target_col},
                       title=f"{target_col} — Full Series")
    fig_prev.update_traces(line_color="#4B9BFF")
    fig_prev.update_layout(height=320, margin=dict(t=40, b=20))
    st.plotly_chart(fig_prev, use_container_width=True)

    # Completion banner (visible right after chart, before scroll)
    if run_btn:
        st.success("✅ Forecast complete! Scroll down to see results ↓")
        st.markdown(
            "<div style='text-align:center;font-size:1.3rem;animation:bounce 1s infinite;'>"
            "👇 Charts & metrics are below</div>"
            "<style>@keyframes bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(6px)}}</style>",
            unsafe_allow_html=True
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 3 — Forecast results
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🔮 Forecast Results")

    if not run_btn:
        st.info("Configure settings above and click **🚀 Run Forecast** to generate results.")
        return

    if not any([run_hw, run_ar, run_rf]):
        st.warning("Select at least one model.")
        return

    split = int(len(series) * train_pct / 100)
    if split < 24 or (len(series) - split) < 6:
        st.error("Not enough data for this split. Lower train % or use more data.")
        return

    train, test = series.iloc[:split], series.iloc[split:]
    st.caption(f"Split: **{len(train)}** train · **{len(test)}** test periods")

    predictions  = {}
    metrics_rows = []
    n_models = sum([run_hw, run_ar, run_rf])
    prog     = st.progress(0)
    status   = st.empty()
    step     = 0

    if run_hw:
        status.text("Training Holt-Winters...")
        try:
            tp, fp = run_holt_winters(train, test, fh)
            predictions["Holt-Winters"] = (tp, fp)
            metrics_rows.append({"Model": "Holt-Winters", **compute_metrics(test, tp)})
        except Exception as e:
            st.warning(f"Holt-Winters failed: {e}")
        step += 1; prog.progress(step / n_models)

    if run_ar:
        status.text("Training ARIMA...")
        try:
            tp, fp = run_arima(train, test, fh)
            predictions["ARIMA"] = (tp, fp)
            metrics_rows.append({"Model": "ARIMA", **compute_metrics(test, tp)})
        except Exception as e:
            st.warning(f"ARIMA failed: {e}")
        step += 1; prog.progress(step / n_models)

    if run_rf:
        status.text("Training Random Forest...")
        try:
            tp, fp = run_random_forest(train, test, fh)
            predictions["Random Forest"] = (tp, fp)
            metrics_rows.append({"Model": "Random Forest", **compute_metrics(test, tp)})
        except Exception as e:
            st.warning(f"Random Forest failed: {e}")
        step += 1; prog.progress(step / n_models)

    prog.progress(1.0)
    status.empty()

    if not predictions:
        st.error("All models failed. Check your data format.")
        return

    # Forecast plot (left) + metrics & table (right)
    plot_col, metric_col = st.columns([2, 1], gap="large")

    with plot_col:
        st.markdown("#### 📈 Actual vs Forecast")
        st.plotly_chart(make_forecast_plot(train, test, predictions), use_container_width=True)

    with metric_col:
        st.markdown("#### 📐 Model Comparison")
        metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
        def highlight_min(s):
            return ["background-color:#1a4a1a;color:#88ff88"
                    if v == s.min() else "" for v in s]
        st.dataframe(metrics_df.style.apply(highlight_min).format("{:.4f}"),
                     use_container_width=True)
        st.caption("🟢 Green = best per metric")

        st.markdown("#### 🔮 Future Values")
        future_df = pd.DataFrame(
            {name: vals[1].values for name, vals in predictions.items()},
            index=list(predictions.values())[0][1].index
        )
        future_df.index.name = "Date"
        st.dataframe(future_df.round(2), use_container_width=True)
        st.download_button("📥 Download CSV",
                           data=future_df.reset_index().to_csv(index=False),
                           file_name="forecast_output.csv", mime="text/csv")

    st.divider()
    st.caption("👩‍🎓 Sierra Reynolds · Utah State University · Huntsman School of Business")


if __name__ == "__main__":
    main()