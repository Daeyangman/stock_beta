import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os

# 페이지 설정
st.set_page_config(
    page_title="거래량 기반 매매 전략",
    page_icon="📈",
    layout="wide",
)

# 제목 및 설명
st.title('거래량 기반 매매 전략 분석')
st.markdown("""
이 앱은 거래량 기반의 매매 전략을 테스트하고 분석하기 위한 도구입니다. 
거래량 증가와 가격 상승을 결합한 지표를 통해 매수 시점을 찾습니다.
""")

# 리샘플링 파일 경로 설정
resampling_dir = r"C:\Users\ha862\work_space\stock_prediction\data\resampling"

# 사용 가능한 리샘플링 파일 매핑
resampling_files = {
    '5분': '5m.parquet',
    '10분': '10m.parquet',
    '30분': '30m.parquet',
    '1시간': '1h.parquet',
    '2시간': '2h.parquet'
}

# 사이드바 - 파일 선택 및 전략 파라미터 설정
st.sidebar.header('데이터 및 파라미터 설정')

# 리샘플링 데이터 선택
selected_timeframe = st.sidebar.selectbox(
    '시간 단위',
    options=list(resampling_files.keys()),
    index=3  # 기본값: 1시간
)

# 분석 대상 범위 설정
analysis_scope = st.sidebar.radio(
    "분석 대상 범위",
    ["전체 종목", "특정 종목만"],
    index=1  # 기본값: 특정 종목만
)

# 선택한 파일 경로
selected_file = os.path.join(resampling_dir, resampling_files[selected_timeframe])

# 데이터 로드 함수
def load_data(file_path):
    """리샘플링된 파일을 로드합니다."""
    if 'loaded_data' in st.session_state and st.session_state['loaded_file'] == file_path:
        return st.session_state['loaded_data']
    
    try:
        if os.path.exists(file_path):
            with st.spinner(f'{selected_timeframe} 데이터를 로드 중입니다...'):
                df = pd.read_parquet(file_path)
                
                # minute 열이 datetime 형식이 아니면 변환
                if not pd.api.types.is_datetime64_any_dtype(df['minute']):
                    df['minute'] = pd.to_datetime(df['minute'])
                
                # 날짜 열 추가
                df['date'] = df['minute'].dt.date
                
                # 결과 저장 및 반환
                st.session_state['loaded_data'] = df
                st.session_state['loaded_file'] = file_path
                return df
        else:
            st.error(f"파일을 찾을 수 없습니다: {file_path}")
            return None
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

# 다시 로드 버튼
if st.sidebar.button('데이터 로드/새로고침'):
    if 'loaded_data' in st.session_state:
        del st.session_state['loaded_data']
    if 'loaded_file' in st.session_state:
        del st.session_state['loaded_file']
    if 'signals_df' in st.session_state:
        del st.session_state['signals_df']

# 데이터 로드
data = load_data(selected_file)

if data is None:
    st.error(f"{selected_file} 파일을 로드할 수 없습니다. 다른 시간 단위를 선택하거나 파일 경로를 확인하세요.")
    st.stop()
else:
    st.success(f"{selected_timeframe} 데이터 로드 완료: {len(data)} 행, {data['symbol'].nunique()} 종목")

# 특정 종목만 분석할 경우
if analysis_scope == "특정 종목만":
    available_symbols = sorted(data['symbol'].unique())
    selected_symbol_for_analysis = st.sidebar.selectbox(
        "분석할 종목 선택",
        options=available_symbols,
        index=0 if available_symbols else None
    )
    
    if selected_symbol_for_analysis:
        # 선택한 종목만 필터링
        data = data[data['symbol'] == selected_symbol_for_analysis]
        st.success(f"'{selected_symbol_for_analysis}' 종목만 분석합니다. 데이터: {len(data)} 행")

# 최근 거래량 기준 기간(일) (드롭다운)
lookback_options = {f'{i}일': i for i in [1, 3, 5, 7, 10, 15, 20, 30]}
selected_lookback = st.sidebar.selectbox(
    '최근 거래량 기준 기간',
    options=list(lookback_options.keys()),
    index=2  # 기본값: 5일
)
lookback_period = lookback_options[selected_lookback]

# 거래량 중위값 대비 비율(%) (드롭다운)
volume_percentile_options = {f'{i} percentile': i for i in [50, 60, 70, 80, 90, 95]}
selected_volume_percentile = st.sidebar.selectbox(
    '거래량 기준 percentile',
    options=list(volume_percentile_options.keys()),
    index=2  # 기본값: 70 percentile
)
volume_percentile = volume_percentile_options[selected_volume_percentile]

# 거래량 계산 방식 (드롭다운)
volume_calc_methods = {
    '일별 총 거래량 기준': 'total',
    '동일 시점까지의 누적 거래량 기준': 'cumulative'
}
selected_calc_method = st.sidebar.selectbox(
    '거래량 계산 방식',
    options=list(volume_calc_methods.keys()),
    index=0
)
volume_calc_method = volume_calc_methods[selected_calc_method]

# 수익률 계산 기준 선택 (드롭다운)
return_basis_options = {
    '당일 종가 기준': 'close',
    '매수시점 이후 고가 기준': 'high'
}
selected_return_basis = st.sidebar.selectbox(
    '수익률 계산 기준',
    options=list(return_basis_options.keys()),
    index=0
)
return_basis = return_basis_options[selected_return_basis]

# 거래량 분석 및 매수 시그널 생성 함수
def generate_signals(df, lookback_days, volume_ratio, calc_method='total', return_basis='close'):
    """설정된 조건에 따라 매수 시그널을 생성합니다."""
    if df is None or df.empty:
        return pd.DataFrame()
    
    signals = []
    unique_symbols = df['symbol'].unique()
    
    with st.spinner('매수 시그널을 생성 중입니다...'):
        for symbol in unique_symbols:
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) < 2:  # 최소 2개의 데이터 포인트 필요
                continue
            
            # 날짜만 추출 (이미 추가됨)
            # symbol_data['date'] = symbol_data['minute'].dt.date
            
            # 일자별로 그룹화 - 데이터가 있는 날짜만 포함
            # 거래일만 추출 (실제 데이터가 있는 날짜)
            trading_days = symbol_data['date'].unique()
            trading_days.sort()  # 날짜 정렬
            
            if len(trading_days) < lookback_days + 1:  # 기준 기간 + 현재일 필요
                continue
            
            # 각 거래일에 대해
            for day_idx in range(lookback_days, len(trading_days)):
                current_day = trading_days[day_idx]
                current_day_data = symbol_data[symbol_data['date'] == current_day]
                
                # 직전 N 거래일 데이터 (실제 거래가 있었던 날만 계산)
                reference_days = trading_days[day_idx-lookback_days:day_idx]
                
                for idx, current_row in current_day_data.iterrows():
                    current_time = current_row['minute']
                    
                    # 현재 시간
                    current_hour = current_time.hour
                    current_minute = current_time.minute
                    
                    # 거래량 계산 방식에 따라 다른 로직 적용
                    if calc_method == 'total':
                        # 일별 총 거래량 기준 (장 운영시간으로 나눔)
                        reference_volumes = []
                        for ref_day in reference_days:
                            day_data = symbol_data[symbol_data['date'] == ref_day]
                            if not day_data.empty:
                                total_volume = day_data['volume'].sum()
                                # 6시간으로 나눈 평균 시간당 거래량
                                avg_hourly_volume = total_volume / 6
                                reference_volumes.append(avg_hourly_volume)
                        
                        # 현재 거래량
                        current_volume = current_row['volume']
                        
                    else:  # 동일 시점까지의 누적 거래량 기준
                        reference_volumes = []
                        for ref_day in reference_days:
                            day_data = symbol_data[symbol_data['date'] == ref_day]
                            if not day_data.empty:
                                # 현재 시간까지의 누적 거래량 계산
                                # 날짜와 시간을 추출하여 문자열로 변환
                                day_str = ref_day.strftime("%Y-%m-%d")
                                time_str = f"{current_hour:02d}:{current_minute:02d}:00"
                                time_limit_str = f"{day_str} {time_str}"
                                
                                # 문자열 비교 방식 사용
                                day_data_until_time = day_data[day_data['minute'].dt.strftime("%Y-%m-%d %H:%M:%S") <= time_limit_str]
                                day_volume_until_time = day_data_until_time['volume'].sum()
                                reference_volumes.append(day_volume_until_time)
                        
                        # 현재 시간까지의 거래량
                        # 날짜와 시간을 추출하여 문자열로 변환
                        day_str = current_day.strftime("%Y-%m-%d")
                        time_str = f"{current_hour:02d}:{current_minute:02d}:00"
                        time_limit_str = f"{day_str} {time_str}"
                        
                        # 문자열 비교 방식 사용
                        current_day_data_until_time = symbol_data[
                            (symbol_data['date'] == current_day) & 
                            (symbol_data['minute'].dt.strftime("%Y-%m-%d %H:%M:%S") <= time_limit_str)
                        ]
                        current_volume = current_day_data_until_time['volume'].sum()
                    
                    if not reference_volumes:  # 참조 거래량이 없으면 건너뜀
                        continue
                        
                    # 참조 거래량의 percentile 값 직접 계산
                    volume_threshold = np.percentile(reference_volumes, volume_percentile)
                    
                    # 거래량 기준값이 0인 경우 건너뜀
                    if volume_threshold == 0:
                        continue
                    
                    # 직전 거래일의 종가 찾기 (날짜 인덱스 기준)
                    if day_idx > 0:
                        prev_trading_day = trading_days[day_idx-1]  # 직전 거래일
                        prev_day_data = symbol_data[symbol_data['date'] == prev_trading_day]
                        if not prev_day_data.empty:
                            prev_day_close = prev_day_data.iloc[-1]['close_price']
                        else:
                            continue
                    else:
                        continue
                    
                    # 거래량 조건 및 가격 조건 확인
                    if (current_volume > volume_threshold and 
                        current_row['close_price'] > prev_day_close):
                        
                        # 해당 시점 이후의 데이터로 수익률 계산
                        future_data = symbol_data[
                            (symbol_data['date'] == current_day) & 
                            (symbol_data['minute'] > current_time)
                        ]
                        
                        # 시그널 이후 첫 번째 봉의 시가를 매수가로 설정
                        if not future_data.empty:
                            # 시그널 직후 첫 번째 봉의 시가 가져오기
                            next_candle_open = future_data.iloc[0]['open_price']
                            entry_price = next_candle_open  # 실제 매수가
                            
                            # 종가 및 최고가 계산
                            day_close = symbol_data[symbol_data['date'] == current_day].iloc[-1]['close_price']
                            day_high_after_signal = future_data['high_price'].max()
                            
                            # 매수 시점 가격 대비 수익률
                            if entry_price != 0:  # 0으로 나누기 방지
                                close_return = (day_close / entry_price - 1) * 100
                                max_return = (day_high_after_signal / entry_price - 1) * 100
                            else:
                                close_return = 0
                                max_return = 0
                        else:
                            # 해당 날짜에 이후 데이터가 없는 경우(마지막 봉) - 시그널 무시 또는 현재 종가를 사용
                            continue  # 이 경우 시그널을 무시 (다음 봉이 없으므로 매수 불가)
                        
                        # 선택한 수익률 계산 기준에 따라 분석 수익률 설정
                        primary_return = close_return if return_basis == 'close' else max_return
                        
                        # 시그널 정보 저장
                        signal_info = {
                            'symbol': symbol,
                            'date': current_day,
                            'time': current_time,
                            'signal_price': current_row['close_price'],  # 시그널 발생 가격
                            'entry_price': entry_price,  # 실제 매수가 (다음 봉 시가)
                            'volume': current_volume,
                            'volume_threshold': volume_threshold,
                            'volume_ratio': (current_volume / volume_threshold * 100) if volume_threshold != 0 else 0,  # 0으로 나누기 방지
                            'prev_day_close': prev_day_close,
                            'prev_trading_day': prev_trading_day,  # 직전 거래일 추가
                            'day_close': day_close,  # 당일 종가 추가
                            'high_after_signal': day_high_after_signal,  # 시그널 이후 최고가
                            'close_return': close_return,
                            'max_return': max_return,  # 일중 최대 수익률 (매수 시점부터 당일 최고가까지)
                            'primary_return': primary_return,  # 선택한 기준에 따른 주요 수익률
                            'return_basis': return_basis  # 분석에 사용된 수익률 기준
                        }
                        signals.append(signal_info)
    
    return pd.DataFrame(signals) if signals else pd.DataFrame()

# 시그널 분석 버튼
if st.sidebar.button('매수 시그널 분석 실행'):
    start_time = datetime.datetime.now()
    signals_df = generate_signals(data, lookback_period, volume_percentile, volume_calc_method, return_basis)
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    
    # 이전 시그널 결과 삭제 및 새 결과 저장
    st.session_state['signals_df'] = signals_df
    
    if len(signals_df) > 0:
        st.success(f"분석 완료: {len(signals_df)}개의 매수 시그널 발견, 소요 시간: {elapsed_time:.2f}초")
    else:
        st.warning("설정된 조건에 맞는 매수 시그널이 발견되지 않았습니다. 파라미터를 조정해보세요.")
else:
    # 이전에 생성된 시그널이 있는지 확인
    signals_df = st.session_state.get('signals_df', None)

# 메인 탭
tabs = st.tabs(["시그널 분석 결과", "종목별 상세 분석"])

with tabs[0]:
    if signals_df is None:
        st.info("사이드바에서 '매수 시그널 분석 실행' 버튼을 클릭하세요.")
    elif len(signals_df) == 0:
        st.warning("설정된 조건에 맞는 매수 시그널이 발견되지 않았습니다. 파라미터를 조정해보세요.")
    else:
        # 거래량 기준값이 0인 항목 제거 (추가 안전장치)
        if 'volume_threshold' in signals_df.columns:
            signals_df = signals_df[signals_df['volume_threshold'] > 0]
            
        # 결과 통계
        st.subheader('매수 시그널 분석 결과')
        st.write(f"총 {len(signals_df)}개의 매수 시그널이 발견되었습니다.")
        
        # 수익률 통계
        avg_close_return = signals_df['close_return'].mean()
        avg_max_return = signals_df['max_return'].mean()
        avg_primary_return = signals_df['primary_return'].mean()
        
        positive_close = (signals_df['close_return'] > 0).mean() * 100
        positive_max = (signals_df['max_return'] > 0).mean() * 100
        positive_primary = (signals_df['primary_return'] > 0).mean() * 100
        
        # 선택된 수익률 계산 기준 확인
        return_basis_display = signals_df['return_basis'].iloc[0] if len(signals_df) > 0 else 'close'
        basis_name = "종가" if return_basis_display == 'close' else "매수 후 고가"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"평균 {basis_name} 수익률", f"{avg_primary_return:.2f}%")
        with col2:
            st.metric(f"{basis_name} 기준 승률", f"{positive_primary:.1f}%")
        with col3:
            if return_basis_display == 'close':
                st.metric("평균 매수 후 고가 수익률", f"{avg_max_return:.2f}%")
            else:
                st.metric("평균 종가 수익률", f"{avg_close_return:.2f}%")
        
        # 수익률 분포 그래프
        # 선택된 수익률 기준에 따라 타이틀 설정
        primary_title = "종가 기준 수익률 분포" if return_basis_display == 'close' else "매수 후 고가 기준 수익률 분포"
        secondary_title = "일중 최대 수익률 분포" if return_basis_display == 'close' else "종가 기준 수익률 분포"
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(primary_title, secondary_title)
        )
        
        # 선택된 수익률 기준에 따라 그래프 표시 순서 변경
        if return_basis_display == 'close':
            fig.add_trace(
                go.Histogram(x=signals_df['close_return'], nbinsx=20, name="종가 수익률"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=signals_df['max_return'], nbinsx=20, name="일중 최대 수익률"),
                row=1, col=2
            )
        else:
            fig.add_trace(
                go.Histogram(x=signals_df['max_return'], nbinsx=20, name="매수 후 고가 수익률"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(x=signals_df['close_return'], nbinsx=20, name="종가 수익률"),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 매수 시그널 테이블
        st.subheader('매수 시그널 상세 정보')
        
        # 데이터 형식 조정
        display_df = signals_df.copy()
        display_df['date'] = display_df['date'].astype(str)
        display_df['prev_trading_day'] = display_df['prev_trading_day'].astype(str)
        display_df['time'] = display_df['time'].dt.strftime('%H:%M')
        
        # 당일 종가가 없는 경우 추가
        if 'day_close' not in display_df.columns:
            display_df['day_close'] = display_df['close_price']  # 임시로 매수가로 대체
        
        # 선택된 수익률 계산 기준 확인
        return_basis_display = signals_df['return_basis'].iloc[0] if len(signals_df) > 0 else 'close'
        
        # 열 순서 및 이름 정리 - 선택된 수익률 기준에 따라 조정
        if return_basis_display == 'close':
            columns_to_display = [
                'symbol', 'date', 'time', 'signal_price', 'entry_price', 'volume', 
                'volume_threshold', 'volume_ratio', 'prev_day_close', 'prev_trading_day',
                'day_close', 'close_return', 'max_return'
            ]
        else:  # 'high' 기준인 경우
            columns_to_display = [
                'symbol', 'date', 'time', 'signal_price', 'entry_price', 'volume', 
                'volume_threshold', 'volume_ratio', 'prev_day_close', 'prev_trading_day',
                'high_after_signal', 'max_return', 'close_return'
            ]
        
        # 존재하는 컬럼만 선택
        available_columns = [col for col in columns_to_display if col in display_df.columns]
        display_df = display_df[available_columns]
        
        # 컬럼 이름 매핑 - 선택된 수익률 기준에 따라 조정
        column_mapping = {
            'symbol': '종목',
            'date': '날짜',
            'time': '시그널 시간',
            'signal_price': '시그널 가격',
            'entry_price': '매수가(다음봉 시가)',
            'volume': '거래량',
            'volume_threshold': '거래량 기준값',
            'volume_ratio': '거래량 비율(%)',
            'prev_day_close': '전일 종가',
            'prev_trading_day': '직전 거래일',
            'day_close': '당일 종가',
            'high_after_signal': '매수 후 고가',
            'close_return': '종가 수익률(%)',
            'max_return': '고가 수익률(%)'
        }
        
        # 존재하는 컬럼만 이름 변경
        display_df.columns = [column_mapping[col] for col in available_columns]
        
        # 소수점 및 형식 정리
        for col in ['거래량 비율(%)', '종가 수익률(%)', '일중 최대 수익률(%)']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        for col in ['거래량', '거래량 기준값']:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(int)
        
        st.dataframe(display_df, use_container_width=True)

with tabs[1]:
    if signals_df is None or len(signals_df) == 0:
        st.info("먼저 매수 시그널 분석을 실행하세요.")
    else:
        # 종목별 시각화
        st.subheader('종목별 상세 분석')
        
        selected_symbol = st.selectbox(
            '종목 선택', 
            options=signals_df['symbol'].unique()
        )
        
        if selected_symbol and data is not None:
            # 해당 종목 데이터 필터링
            symbol_data = data[data['symbol'] == selected_symbol].copy()
            symbol_signals = signals_df[signals_df['symbol'] == selected_symbol].copy()
            
            if not symbol_data.empty:
                # 시계열 그래프에서 없는 날짜 제거 (거래가 없는 날)
                symbol_data = symbol_data.sort_values('minute')
                
                # 캔들스틱 차트 + 거래량 + 시그널 마킹
                fig = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"{selected_symbol} 가격 차트 ({selected_timeframe})", "거래량")
                )
                
                # 캔들스틱 추가 - 거래가 있는 날만 표시
                fig.add_trace(
                    go.Candlestick(
                        x=symbol_data['minute'],
                        open=symbol_data['open_price'], 
                        high=symbol_data['high_price'],
                        low=symbol_data['low_price'], 
                        close=symbol_data['close_price'],
                        name="가격"
                    ),
                    row=1, col=1
                )
                
                # 거래량 바 추가
                fig.add_trace(
                    go.Bar(
                        x=symbol_data['minute'],
                        y=symbol_data['volume'],
                        name="거래량",
                        marker_color='rgba(0, 0, 255, 0.5)'
                    ),
                    row=2, col=1
                )
                
                # 시그널 마킹
                if not symbol_signals.empty:
                    # 시그널 시점 표시
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_signals['time'],
                            y=symbol_signals['signal_price'],
                            mode='markers',
                            name='매수 시그널',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color='green',
                                line=dict(width=2, color='darkgreen')
                            )
                        ),
                        row=1, col=1
                    )
                    
                    # 실제 매수 시점 표시 (다음 봉 시가)
                    fig.add_trace(
                        go.Scatter(
                            x=symbol_signals['time'],  # 시간은 시그널과 동일하게 표시 (시각적 연결)
                            y=symbol_signals['entry_price'],
                            mode='markers',
                            name='실제 매수가',
                            marker=dict(
                                symbol='circle',
                                size=8,
                                color='red',
                                line=dict(width=2, color='darkred')
                            )
                        ),
                        row=1, col=1
                    )
                
                # 레이아웃 업데이트
                fig.update_layout(
                    height=700,
                    xaxis_rangeslider_visible=False,
                    showlegend=True
                )
                
                # x축 설정: 거래가 없는 날짜는 간격을 없앰
                fig.update_xaxes(
                    rangebreaks=[
                        # 주말 설정 (거래가 없는 날은 자동으로 건너뜀)
                        dict(pattern="day of week", bounds=[6, 1])  # 토요일과 일요일
                    ]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 선택된 종목의 시그널 정보
                if not symbol_signals.empty:
                    st.write(f"{selected_symbol}의 매수 시그널 ({len(symbol_signals)}개)")
                    
                    # 표시 형식 조정
                    symbol_display = symbol_signals.copy()
                    symbol_display['date'] = symbol_display['date'].astype(str)
                    symbol_display['prev_trading_day'] = symbol_display['prev_trading_day'].astype(str)
                    symbol_display['time'] = symbol_display['time'].dt.strftime('%H:%M')
                    
                    # 당일 종가가 없는 경우 추가
                    if 'day_close' not in symbol_display.columns:
                        symbol_display['day_close'] = symbol_display['close_price']  # 임시로 매수가로 대체
                    
                    # 선택된 수익률 계산 기준 확인
                    return_basis_display = symbol_signals['return_basis'].iloc[0] if len(symbol_signals) > 0 else 'close'
                    
                    # 열 순서 및 이름 정리 - 선택된 수익률 기준에 따라 조정
                    if return_basis_display == 'close':
                        columns_to_display = [
                            'date', 'time', 'signal_price', 'entry_price', 'volume', 
                            'volume_threshold', 'volume_ratio', 'prev_day_close', 'prev_trading_day',
                            'day_close', 'close_return', 'max_return'
                        ]
                    else:  # 'high' 기준인 경우
                        columns_to_display = [
                            'date', 'time', 'signal_price', 'entry_price', 'volume', 
                            'volume_threshold', 'volume_ratio', 'prev_day_close', 'prev_trading_day',
                            'high_after_signal', 'max_return', 'close_return'
                        ]
                    
                    # 존재하는 컬럼만 선택
                    available_columns = [col for col in columns_to_display if col in symbol_display.columns]
                    symbol_display = symbol_display[available_columns]
                    
                    # 컬럼 이름 매핑 - 선택된 수익률 기준에 따라 조정
                    column_mapping = {
                        'date': '날짜',
                        'time': '시그널 시간',
                        'signal_price': '시그널 가격',
                        'entry_price': '매수가(다음봉 시가)',
                        'volume': '거래량',
                        'volume_threshold': '거래량 기준값',
                        'volume_ratio': '거래량 비율(%)',
                        'prev_day_close': '전일 종가',
                        'prev_trading_day': '직전 거래일',
                        'day_close': '당일 종가',
                        'high_after_signal': '매수 후 고가',
                        'close_return': '종가 수익률(%)',
                        'max_return': '고가 수익률(%)'
                    }
                    
                    # 존재하는 컬럼만 이름 변경
                    symbol_display.columns = [column_mapping[col] for col in available_columns]
                    
                    for col in ['거래량 비율(%)', '종가 수익률(%)', '일중 최대 수익률(%)']:
                        if col in symbol_display.columns:
                            symbol_display[col] = symbol_display[col].round(2)
                    
                    for col in ['거래량', '거래량 기준값']:
                        if col in symbol_display.columns:
                            symbol_display[col] = symbol_display[col].astype(int)
                    
                    st.dataframe(symbol_display, use_container_width=True)
            else:
                st.write(f"{selected_symbol}에 대한 데이터가 없습니다.")

# 추가 정보
st.sidebar.markdown("""
### 전략 설명
1. 최근 N 거래일간의 거래량 중 사용자가 선택한 percentile 값을 초과하는 거래량 발생
2. 직전 거래일의 종가보다 현재 가격이 높음
3. 두 조건이 모두 충족되면 매수 시그널 생성
""")

st.sidebar.markdown("""
### 거래량 계산 방식
- **일별 총 거래량 기준**: 일일 총 거래량을 장 운영시간(6시간)으로 나눈 평균 시간당 거래량을 사용
- **동일 시점까지의 누적 거래량 기준**: 각 일자별로 현재 시간까지 누적된 거래량을 비교

### 수익률 계산 기준
- **당일 종가 기준**: 매수시점부터 당일 장 마감까지의 가격 변화율 (안정적인 수익)
- **매수시점 이후 고가 기준**: 매수시점부터 당일 장중 최고가까지의 가격 변화율 (최대 가능 수익)
""")