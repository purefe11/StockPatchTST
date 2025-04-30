import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def draw_sin_cos_encoding(labels, title, width=800, height=400):
    count = len(labels)
    x_values = np.arange(1, count + 1)

    # Sin-Cos Encoding
    sin_values = np.sin(2 * np.pi * x_values / count)
    cos_values = np.cos(2 * np.pi * x_values / count)

    # Plotly 그래프 생성
    fig = go.Figure()

    # Sin 그래프 추가
    fig.add_trace(go.Scatter(
        x=x_values, y=sin_values,
        mode='lines+markers',
        name='Sin Encoding',
        line=dict(color='blue')
    ))

    # Cos 그래프 추가
    fig.add_trace(go.Scatter(
        x=x_values, y=cos_values,
        mode='lines+markers',
        name='Cos Encoding',
        line=dict(color='red')
    ))

    # 그래프 레이아웃 설정
    fig.update_layout(
        title='Sin-Cos Encoding of Months',
        xaxis=dict(title=title, tickmode='array', tickvals=x_values, ticktext=labels),
        yaxis=dict(title='Value'),
        # template='plotly_dark',
        # legend=dict(x=0, y=1),
        width=width,
        height=height,
    )

    # 그래프 출력
    fig.show()


def show_histogram(data, title, width=400, height=300):
    # 데이터 개수를 미리 집계
    hist, bin_edges = np.histogram(data, bins=30)

    # 히스토그램 시각화
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bin_edges[:-1], y=hist, width=np.diff(bin_edges)))

    # 레이아웃 설정
    fig.update_layout(
        xaxis_title=title,
        yaxis_title="",
        bargap=0.1,
        width=width,
        height=height
    )

    # 마우스 오버 시 툴팁 제거
    # fig.update_traces(hoverinfo="skip", hovertemplate=None)

    # Mode Bar(우측 상단 메뉴) 제거
    fig.show()


def show_heatmap(df, feature_columns, width=2000, height=2000):
    corr_matrix = df[feature_columns].corr()

    # 각 셀에 표시할 상관계수 값을 문자열로 변환
    text_matrix = np.round(corr_matrix.values, 2).astype(str)

    # 사용자 정의 컬러맵
    custom_colorscale = [
        [0, "blue"],  # -1 → 파랑
        [0.5, "white"],  # 0 → 흰색
        [1, "red"]  # 1 → 빨강
    ]

    # Plotly 히트맵 생성 (셀 내부에 값 추가)
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,  # 상관계수 값
        x=corr_matrix.columns,  # X축 레이블
        y=corr_matrix.index,  # Y축 레이블
        colorscale=custom_colorscale,
        zmin=-1,  # 최소값
        zmax=1,  # 최대값
        text=text_matrix,  # 셀 내부에 표시할 값
        texttemplate="%{text}",  # 셀 내부에 값 표시
        hoverinfo="text",  # 마우스 오버 시 값 표시
        showscale=True,  # 컬러바 표시
        xgap=1,  # X축 간격 (경계선 효과)
        ygap=1  # Y축 간격 (경계선 효과)
    ))

    # 그래프 레이아웃 조정
    fig.update_layout(
        title="📊 Feature Correlation Heatmap",
        xaxis=dict(tickmode="array", tickvals=np.arange(len(corr_matrix.columns)), ticktext=corr_matrix.columns),
        yaxis=dict(tickmode="array", tickvals=np.arange(len(corr_matrix.index)), ticktext=corr_matrix.index,
                   autorange="reversed"),
        width=width,
        height=height,
    )

    # 마우스 오버 시 툴팁 제거
    fig.update_traces(hoverinfo="skip", hovertemplate=None)

    # 그래프 표시
    fig.show()


def show_feature_importance(feature_importance, width=1200, height=1000):
    # 피처 중요도
    feature_importance = pd.DataFrame({
        "Feature": list(feature_importance.keys()),
        "Importance": list(feature_importance.values())
    }).sort_values(by="Importance", ascending=False)

    # 피처 중요도 그래프 생성
    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="📊 Feature Importance",
        labels={"Importance": "Feature Importance", "Feature": "Feature"},
        color="Importance",
        color_continuous_scale="blues"
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),  # 중요도가 높은 피처가 위로 오도록 정렬
        xaxis_title="Feature Importance",
        yaxis_title="Feature",
        margin=dict(l=150, r=50, t=50, b=50),  # 좌측 마진 추가
        width=width,
        height=height,
    )
    fig.show(config={'staticPlot': True})


def show_regression(y_test_reg, y_pred_reg, width=800, height=400):
    # 실제 vs 예측 비교 (산점도)
    fig1 = px.scatter(
        x=y_test_reg, y=y_pred_reg,
        labels={"x": "실제 상승률 (%)", "y": "예측된 상승률 (%)"},
        title="📊 실제 vs 예측 상승률 비교",
        trendline="ols"
    )
    fig1.add_trace(go.Scatter(x=[min(y_test_reg), max(y_test_reg)],
                              y=[min(y_test_reg), max(y_test_reg)],
                              mode="lines", name="완벽한 예측", line=dict(color="red", dash="dot")))
    fig1.update_layout(
        width=width,
        height=height
    )
    fig1.show()

    # 잔차 분석 (Residual Plot)
    residuals = y_test_reg - y_pred_reg
    fig2 = px.scatter(
        x=y_test_reg, y=residuals,
        labels={"x": "실제 상승률 (%)", "y": "잔차 (Residual)"},
        title="📉 잔차 분석 (Residual Plot)"
    )
    fig2.add_hline(y=0, line_dash="dot", line_color="red")
    fig2.update_layout(
        width=width,
        height=height
    )
    fig2.show()

    # 예측값 vs 실제값 분포 비교 (히스토그램)
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=y_test_reg, name="실제 상승률", opacity=0.6))
    fig3.add_trace(go.Histogram(x=y_pred_reg, name="예측된 상승률", opacity=0.6))
    fig3.update_layout(
        title="📈 실제 vs 예측 상승률 분포 비교",
        xaxis_title="상승률 (%)",
        barmode="overlay",
        template="plotly_white",
        width=width,
        height=height
    )
    fig3.show()


def show_confusion_matrix(confusion_matrix, labels, width=600, height=500):
    # Plotly Confusion Matrix 시각화
    fig = ff.create_annotated_heatmap(
        z=confusion_matrix,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True
    )
    fig.update_layout(
        title="Confusion Matrix (Plotly)",
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        template="plotly_white",
        width=width,
        height=height,
    )
    fig.show(config={'staticPlot': True})


def draw_month_roi(df, width=1200, height=600):
    year = df.head(1)['year'].item()

    # 월별 성공 빈도 계산
    df_grouped = df.groupby('month').agg(
        buy_signal=('buy_signal', 'sum'),
        success=('success', 'sum')
    ).reset_index()

    # 데이터 변환 (melt)
    df_grouped['month'] = df_grouped['month'].astype(str)
    df_melted = df_grouped.melt(id_vars=['month'], var_name='Category', value_name='Value')
    df_melted['Category'] = df_melted['Category'].map({'buy_signal': '매수', 'success': "성공"})

    # Plotly Bar Chart
    fig = px.bar(df_melted, x='month', y='Value', color='Category',
                 barmode='group', text=None,
                 labels={'month': '월', 'Value': '빈도', 'Category': '구분'},
                 title=f"📊 {year} (월별)",
                 color_discrete_sequence=px.colors.qualitative.Safe,
                 category_orders={"month": list(range(1, 13))}
                 )

    # 그래프 스타일 조정
    fig.update_layout(
        # yaxis=dict(
        #     tickmode='linear',
        #     # dtick=1  # y축 간격을 1로 설정 (정수 간격)
        # ),
        xaxis={'type': 'category'},
        xaxis_title="Month",
        yaxis_title="Frequency",
        legend_title="",
        bargap=0.2,
        width=width,
        height=height,
        template="plotly_white"
    )

    fig.show()


def draw_daily_roi(df, month, width=1200, height=600):
    year = df.head(1)['year'].item()

    df = df.copy()
    df['day'] = df['datetime'].dt.day

    # 일별 성공 빈도 계산
    df_grouped = df[df['month'] == month].groupby('day').agg(
        buy_signal=('buy_signal', 'sum'),
        success=('success', 'sum')
    ).reset_index()

    # 데이터 변환 (melt)
    df_grouped['day'] = df_grouped['day'].astype(str)
    df_melted = df_grouped.melt(id_vars=['day'], var_name='Category', value_name='Value')
    df_melted['Category'] = df_melted['Category'].map({'buy_signal': '매수', 'success': "성공"})

    # Plotly Bar Chart
    fig = px.bar(df_melted, x='day', y='Value', color='Category',
                 barmode='group', text=None,
                 labels={'day': '월', 'Value': '빈도', 'Category': '구분'},
                 title=f"📊 {year}년 {month}월 (일별)",
                 color_discrete_sequence=px.colors.qualitative.Safe,
                 category_orders={"day": list(range(1, 13))}
                 )

    # 그래프 스타일 조정
    fig.update_layout(
        # yaxis=dict(
        #     tickmode='linear',
        #     # dtick=1  # y축 간격을 1로 설정 (정수 간격)
        # ),
        xaxis={'type': 'category'},
        xaxis_title="day",
        yaxis_title="Frequency",
        legend_title="",
        bargap=0.2,
        width=width,
        height=height,
        template="plotly_white"
    )

    fig.show()


def draw_group_data(df, group, columns, agg=('min', 'max'), title='', width=1400, height=600):
    # 데이터 그룹화 및 집계
    df = df.groupby(group)[columns].agg(agg)

    # MultiIndex 컬럼을 단순화 (('col', 'min') → 'col_min')
    df.columns = ['_'.join(col) for col in df.columns]

    # 인덱스를 열로 변환
    df.reset_index(inplace=True)

    # 데이터 변형
    df = df.melt(id_vars=[group], var_name='Metric', value_name='Value')

    fig = px.bar(df, x=group, y="Value", color="Metric", barmode="group", title=title, width=width, height=height)

    # 모든 x축 라벨 노출
    fig.update_xaxes(tickmode="array", tickvals=df[group].unique())

    fig.show()


def draw_stock_roi(df, display_count, width=2000, height=800):
    year = df.head(1)['year'].item()

    # 종목별 성공 빈도 계산
    df_grouped = df[df['buy_signal']].groupby('stock_name').agg(
        buy_signal=('buy_signal', 'sum'),
        success=('success', 'sum')
    ).reset_index()
    df_grouped = df_grouped.sort_values(by=['success', 'buy_signal'], ascending=[False, False]).head(display_count)

    # 데이터 변환 (melt)
    df_melted = df_grouped.melt(id_vars=['stock_name'], var_name='Category', value_name='Value')
    df_melted['Category'] = df_melted['Category'].map({'buy_signal': '매수', 'success': "성공"})

    # Plotly Bar Chart
    fig = px.bar(df_melted, x='stock_name', y='Value', color='Category',
                 barmode='group', text=None,
                 labels={'stock_name': '종목', 'Value': '빈도', 'Category': '구분'},
                 title=f"📊 {year} (종목별)",
                 color_discrete_sequence=px.colors.diverging.PiYG)

    # 그래프 스타일 조정
    fig.update_layout(
        xaxis_title="Stock",
        yaxis_title="Frequency",
        legend_title="",
        bargap=0.2,
        width=width,
        height=height,
        template="plotly_white"
    )

    fig.show()


def show_binary_continuous_correlation(df, col1, label1, cols2, label2, width=600, height=400):
    year = df.head(1)['year'].item()

    print(
        f'==========================================================================================\n'
        f'✅ {year}\n'
        f'=========================================================================================='
    )
    # for col2 in cols2:
    #     # Point Biserial Correlation (연속형 & 이진형 상관계수)
    #     # p-value(유의확률: 상관관계가 우연히 발생했을 확률)
    #     corr, p_value = pointbiserialr(df[col1], df[col2])
    #     print(
    #         f'{label1}에 따라 {col2} 상관관계가 통계적으로 {"유의미" if p_value < 0.05 else "무의미"}합니다.\n'
    #         f'Point-Biserial Correlation: {corr:.4f}, p-value: {p_value:.4f}\n'
    #         f'------------------------------------------------------------------------------------------'
    #     )

    df_grouped = df.groupby(col1)[cols2].mean().reset_index()

    # True/False 색상 지정 (레드/블루)
    color_map = {True: "red", False: "blue"}

    # Bar 그래프 (범주별 평균값)
    df_melted = df_grouped.melt(id_vars=col1, value_vars=cols2, var_name='metric', value_name='value')
    fig_bar = px.bar(df_melted, x='metric', y='value', color=col1,
                     barmode='group',  # 그룹 형태
                     color_discrete_map=color_map,
                     title=f"{label1}에 따라 {label2} 평균값")
    fig_bar.update_layout(bargap=0.1, width=width, height=height)
    fig_bar.show()

    # Boxplot 그래프 (Outlier 및 중앙값 확인)
    df_melted = df.melt(id_vars=col1, value_vars=cols2, var_name='metric', value_name='value')
    fig_box = px.box(df_melted, x='metric', y='value', color=col1,
                     color_discrete_map=color_map,
                     points=False, title=f"{label1}에 따라 {label2} 분포")
    fig_box.update_layout(bargap=0.1, width=width, height=height)
    fig_box.show()

    # Histogram (False인 경우, 정규분포 및 편향성 확인)
    df_melted = df[df[col1] == False].melt(id_vars=col1, value_vars=cols2, var_name='metric', value_name='value')
    fig = px.histogram(df_melted, x='value', color=col1, facet_col='metric', facet_col_wrap=2,
        opacity=0.6, nbins=30, color_discrete_map=color_map, title=f"{label1} False일 때 {label2} 분포",)
    fig.add_vline(x=0, line_dash="dash", line_color="black")  # 중앙선 추가
    fig.update_layout(width=width, height=height, bargap=0.1)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.show()

    # Histogram (True인 경우, 정규분포 및 편향성 확인)
    df_melted = df[df[col1] == True].melt(id_vars=col1, value_vars=cols2, var_name='metric', value_name='value')
    fig = px.histogram(df_melted, x='value', color=col1, facet_col='metric', facet_col_wrap=2,
        opacity=0.6, nbins=30, color_discrete_map=color_map, title=f"{label1} True일 때 {label2} 분포",)
    fig.add_vline(x=0, line_dash="dash", line_color="black")  # 중앙선 추가
    fig.update_layout(width=width, height=height, bargap=0.1)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.show()


def show_stock_chart(df_stock, title, min_profit_rate=0.9, width=1800, height=1200):
    # 1년치 전체 날짜 생성 (모든 날 포함)
    all_dates = pd.date_range(start=df_stock["datetime"].min(), end=df_stock["datetime"].max(), freq="D")

    # 거래일 데이터만 남기고 없는 날짜 찾기
    trading_dates = set(df_stock["datetime"])  # 실제 거래일 리스트
    non_trading_days = list(set(all_dates) - trading_dates)  # 거래되지 않은 날짜 리스트

    # Plotly: 캔들 차트 & 이동평균선 & 매수/매도 신호 추가 + 거래량 차트 추가
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.7, 0.15, 0.15], vertical_spacing=0.02)

    # 캔들 차트 추가
    fig.add_trace(go.Candlestick(
        x=df_stock["datetime"],
        open=df_stock["open"],
        high=df_stock["high"],
        low=df_stock["low"],
        close=df_stock["close"],
        name="Candle",
        increasing_line=dict(color="red", width=1),
        decreasing_line=dict(color="blue", width=1)
    ))

    # 250d 신고가 추가
    # fig.add_trace(go.Scatter(
    #     x=df_stock["datetime"], y=df_stock["max_250"],
    #     mode="lines", name="신고가",
    #     line=dict(color="red", dash="dot", width=1)
    # ))

    # # 5일 이동평균선 추가
    # fig.add_trace(go.Scatter(
    #     x=df_stock["datetime"], y=df_stock["ma5"],
    #     mode="lines", name="5MA",
    #     line=dict(color="green", dash="solid", width=1)
    # ))
    #
    # # 20일 이동평균선 추가
    # fig.add_trace(go.Scatter(
    #     x=df_stock["datetime"], y=df_stock["ma20"],
    #     mode="lines", name="20MA",
    #     line=dict(color="orange", dash="solid", width=1)
    # ))

    # 5일 VWMA(거래 가중 이동평균) 추가
    fig.add_trace(go.Scatter(
        x=df_stock["datetime"], y=df_stock["vwma5"],
        mode="lines", name="5VWMA",
        line=dict(color="green", dash="solid", width=1)
    ))

    # 20일 VWMA(거래 가중 이동평균) 추가
    fig.add_trace(go.Scatter(
        x=df_stock["datetime"], y=df_stock["vwma20"],
        mode="lines", name="20VWMA",
        line=dict(color="orange", dash="solid", width=1)
    ))

    # 60일 이동평균선 추가
    # fig.add_trace(go.Scatter(
    #     x=df_stock["datetime"], y=df_stock["ma60"],
    #     mode="lines", name="60MA",
    #     line=dict(color="purple", dash="solid", width=1)
    # ))

    # # 볼린저 밴드 구름 영역 (하한선~상한선 채우기)
    # fig.add_trace(go.Scatter(
    #     x=df_stock["datetime"], y=df_stock["bb_upper"],
    #     mode="lines", name="볼린저 상",
    #     line=dict(color="#FFDBBB", width=1),
    #     fill=None
    # ))
    # fig.add_trace(go.Scatter(
    #     x=df_stock["datetime"], y=df_stock["bb_lower"],
    #     mode="lines", name="볼린저 하",
    #     line=dict(color="#FFDBBB", width=1),
    #     fill="tonexty",
    #     fillcolor="rgba(255, 219, 187, 0.5)"
    # ))

    # VWMA 볼린저 밴드 구름 영역 (하한선~상한선 채우기)
    fig.add_trace(go.Scatter(
        x=df_stock["datetime"], y=df_stock["vwma_bb_upper"],
        mode="lines", name="VWMA 볼린저 상",
        line=dict(color="#FFDBBB", width=1),
        fill=None
    ))
    fig.add_trace(go.Scatter(
        x=df_stock["datetime"], y=df_stock["vwma_bb_lower"],
        mode="lines", name="VWMA 볼린저 하",
        line=dict(color="#FFDBBB", width=1),
        fill="tonexty",
        fillcolor="rgba(255, 219, 187, 0.5)"
    ))

    # 매수 신호 추가
    buy_signals = df_stock[df_stock["buy_signal"]]
    fig.add_trace(go.Scatter(
        x=buy_signals["datetime"], y=buy_signals["close"],
        mode="markers", name="매수",
        marker=dict(color="red", symbol="triangle-up", size=10)
    ))

    # 매도 신호 추가
    sell_signals = df_stock[df_stock["sell_signal"]]
    fig.add_trace(go.Scatter(
        x=sell_signals["datetime"], y=sell_signals["close"],
        mode="markers", name="매도",
        marker=dict(color="blue", size=10, symbol="triangle-down")
    ))

    # 거래량 차트 추가 (막대 그래프)
    fig.add_trace(go.Bar(
        x=df_stock["datetime"],
        y=df_stock["trading_volume"],
        name="거래량",
        marker=dict(color=df_stock.apply(lambda x: "red" if x['trading_volume'] > x['prev_trading_volume'] else "blue", axis=1))
    ), row=2, col=1)

    # 거래량 5일 이동평균선 추가
    fig.add_trace(go.Scatter(
        x=df_stock["datetime"], y=df_stock["vol_ma5"],
        mode="lines", name="V_5MA",
        line=dict(color="green", dash="solid", width=1)
    ), row=2, col=1)

    # 거래량 20일 이동평균선 추가
    fig.add_trace(go.Scatter(
        x=df_stock["datetime"], y=df_stock["vol_ma20"],
        mode="lines", name="V_20MA",
        line=dict(color="red", dash="solid", width=1)
    ), row=2, col=1)

    # 개인 순매수량 차트 추가 (막대 그래프)
    fig.add_trace(go.Bar(
        x=df_stock["datetime"],
        y=df_stock["individual"],
        name="개인",
        marker=dict(color="deepskyblue")
    ), row=3, col=1)

    # 기관 순매수량 차트 추가 (막대 그래프)
    fig.add_trace(go.Bar(
        x=df_stock["datetime"],
        y=df_stock["institution"],
        name="기관",
        marker=dict(color="limegreen")
    ), row=3, col=1)

    # 외국인 순매수량 차트 추가 (막대 그래프)
    fig.add_trace(go.Bar(
        x=df_stock["datetime"],
        y=df_stock["foreign"],
        name="외국인",
        marker=dict(color="tomato")
    ), row=3, col=1)

    # 레이아웃 설정 (가로 스크롤 가능하게 설정)
    fig.update_layout(
        title=f"{title} (종목: {df_stock.iloc[0]['stock_name']})",
        showlegend=True,
        xaxis=dict(
            type="date",
            rangeslider=dict(visible=False),  # 가로 스크롤 바 제거
            rangebreaks=[
                dict(values=non_trading_days)  # 실제 거래일이 아닌 날 제거
            ]
        ),
        yaxis_title="Price",
        # hovermode="x unified",
        # template="plotly_white",
        template="xgridoff",
        barmode="group",
        width=width,
        height=height
    )

    fig.update_layout(
        xaxis=dict(rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=12, label="1y", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )),
    )

    fig.update_yaxes(fixedrange=False)
    fig.show()
