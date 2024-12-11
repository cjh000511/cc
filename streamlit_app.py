import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, timedelta
import pyupbit
import logging
import ta
from autotrade import fibonacci_retracement, analyze_elliott_wave

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_rsi(close_prices, period=14):
    """RSI 계산 함수"""
    try:
        # ta 라이브러리를 사용하여 RSI 계산
        rsi = ta.momentum.RSIIndicator(close_prices, window=period).rsi()
        return rsi.iloc[-1]  # 최신 RSI 값 반환
    except Exception as e:
        logger.error(f"RSI 계산 중 오류: {e}")
        return 50  # 오류 발생시 중립값 반환

def init_db():
    """데이터베이스가 없을 경우 생성"""
    conn = sqlite3.connect('bitcoin_trades.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  decision TEXT,
                  percentage INTEGER,
                  reason TEXT,
                  btc_balance REAL,
                  krw_balance REAL,
                  btc_avg_buy_price REAL,
                  btc_krw_price REAL,
                  rsi REAL,
                  volume REAL,
                  ma5 REAL,
                  ma20 REAL,
                  profit_ratio REAL,
                  trade_amount REAL,
                  
                  /* 피보나치 레벨 */
                  fibo_0 REAL,
                  fibo_236 REAL,
                  fibo_382 REAL,
                  fibo_500 REAL,
                  fibo_618 REAL,
                  fibo_786 REAL,
                  fibo_1 REAL,
                  
                  /* 엘리엇 파동 분석 */
                  elliott_wave_count INTEGER,
                  elliott_wave_direction TEXT,
                  elliott_wave_confidence REAL,
                  elliott_wave_pattern TEXT,
                  
                  reflection TEXT)''')
    conn.commit()
    conn.close()

def get_connection():
    if not os.path.exists('/home/ec2-user/cc/bitcoin_trades.db'):
        init_db()
    return sqlite3.connect('/home/ec2-user/cc/bitcoin_trades.db')

def load_data():
    try:
        conn = get_connection()
        query = """
        SELECT * FROM trades 
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            st.warning("거래 데이터가 아직 없습니다. 자동매매 프로그램을 실행하여 데이터를 생성해주세요.")
            return None
            
        # timestamp 컬럼을 datetime으로 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 비어있는 값 처리
        df = df.fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

def log_monitoring_data(upbit, trend_data=None, entry_data=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 현재 시장 데이터 가져오기
        current_price = pyupbit.get_current_price("KRW-BTC")
        
        # 잔고 정보
        balances = upbit.get_balances()
        btc_balance = float(next((balance['balance'] for balance in balances if balance['currency'] == 'BTC'), 0))
        krw_balance = float(next((balance['balance'] for balance in balances if balance['currency'] == 'KRW'), 0))
        btc_avg_price = float(next((balance['avg_buy_price'] for balance in balances if balance['currency'] == 'BTC'), 0))
        
        # RSI 계산
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute30", count=200)
        rsi = calculate_rsi(df['close'])
        
        # 이동평균선
        ma5 = df['close'].rolling(window=5).mean().iloc[-1]
        ma20 = df['close'].rolling(window=20).mean().iloc[-1]
        
        # 수익률 계산 수정
        total_value = (btc_balance * current_price) + krw_balance
        # 첫 거래 기록이 있는지 확인
        cursor.execute("SELECT krw_balance + (btc_balance * btc_krw_price) as initial_value FROM trades ORDER BY timestamp ASC LIMIT 1")
        result = cursor.fetchone()
        initial_value = result[0] if result else total_value
        
        profit_ratio = ((total_value / initial_value) - 1) * 100 if initial_value > 0 else 0
        
        # 피보나치 레벨 계산
        df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
        fibo_levels = fibonacci_retracement(df)
        
        # 엘리엇 파동 분석
        elliott_analysis = analyze_elliott_wave(df)
        
        # DB에 저장
        cursor.execute('''
            INSERT INTO trades (
                timestamp, decision, percentage, reason,
                btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price,
                rsi, volume, ma5, ma20, profit_ratio, trade_amount,
                fibo_0, fibo_236, fibo_382, fibo_500, fibo_618, fibo_786, fibo_1,
                elliott_wave_count, elliott_wave_direction, elliott_wave_confidence,
                elliott_wave_pattern
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            entry_data.get('decision', 'hold') if entry_data else 'hold',
            entry_data.get('position_size', 0) * 100 if entry_data else 0,
            entry_data.get('reason', '정기 모니터링') if entry_data else '정기 모니터링',
            btc_balance,
            krw_balance,
            btc_avg_price,
            current_price,
            rsi,
            df['volume'].iloc[-1],
            ma5,
            ma20,
            profit_ratio,
            0,
            fibo_levels.get('0', 0),
            fibo_levels.get('0.236', 0),
            fibo_levels.get('0.382', 0),
            fibo_levels.get('0.5', 0),
            fibo_levels.get('0.618', 0),
            fibo_levels.get('0.786', 0),
            fibo_levels.get('1', 0),
            elliott_analysis.get('current_wave', 0),
            elliott_analysis.get('direction', 'unknown'),
            elliott_analysis.get('confidence', 0),
            str(elliott_analysis.get('reason', []))
        ))
        
        conn.commit()
        conn.close()
        logger.info("모니터링 데이터가 DB에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"모니터링 데이터 저장 중 오류: {e}")
        if conn:
            conn.close()

def main():
    st.title('Bitcoin Trading Dashboard')
    
    # 데이터 로드
    df = load_data()
    
    if df is not None and not df.empty:
        # 기간 선택 필터
        st.sidebar.header('기간 설정')
        date_range = st.sidebar.selectbox(
            '조회 기간',
            ['전체', '최근 24시간', '최근 7일', '최근 30일']
        )
        
        # 선택된 기간에 따라 데이터 필터링
        if date_range != '전체':
            now = datetime.now()
            if date_range == '최근 24시간':
                start_date = now - timedelta(days=1)
            elif date_range == '최근 7일':
                start_date = now - timedelta(days=7)
            else:  # 최근 30일
                start_date = now - timedelta(days=30)
            df = df[df['timestamp'] >= start_date]

        # 실시간 시장 상황
        st.header('📊 실시간 시장 상황')
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latest_price = df['btc_krw_price'].iloc[-1]
            price_change = df['btc_krw_price'].pct_change().iloc[-1] * 100
            st.metric("BTC 현재가", 
                     f"{latest_price:,.0f} KRW",
                     f"{price_change:+.2f}%")
            
        with col2:
            latest_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 0
            st.metric("RSI", 
                     f"{latest_rsi:.1f}",
                     "과매수" if latest_rsi > 70 else "과매도" if latest_rsi < 30 else "중립")
            
        with col3:
            # 현재 총 자산 가치
            current_total = (df['btc_balance'].iloc[-1] * df['btc_krw_price'].iloc[-1] + 
                            df['krw_balance'].iloc[-1])
            
            # 초기 투자금 (첫 거래 시점의 총 자산)
            initial_total = (df['btc_balance'].iloc[0] * df['btc_krw_price'].iloc[0] + 
                            df['krw_balance'].iloc[0])
            
            profit_ratio = ((current_total / initial_total) - 1) * 100 if initial_total > 0 else 0
            
            st.metric("수익률", 
                      f"{profit_ratio:+.2f}%",
                      "초기 투자금 대비")
            
        with col4:
            win_rate = (len(df[df['decision'] == 'sell']) / 
                       len(df[df['decision'].isin(['buy', 'sell'])]) * 100)
            st.metric("승률", 
                     f"{win_rate:.1f}%",
                     f"총 {len(df[df['decision'].isin(['buy', 'sell'])])}건")

        # 트레이딩 패턴 분석
        st.header('📈 트레이딩 패턴 분석')
        col1, col2 = st.columns(2)
        
        with col1:
            # 시간대별 거래 분포
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly_trades = df['hour'].value_counts().sort_index()
            fig_hourly = px.bar(x=hourly_trades.index, 
                              y=hourly_trades.values,
                              title='시간대별 거래 분포',
                              labels={'x': '시간', 'y': '거래 횟수'})
            st.plotly_chart(fig_hourly)
            
        with col2:
            # 매수/매도 결정 이유 분석
            reason_counts = df['reason'].value_counts().head(5)
            fig_reasons = px.pie(values=reason_counts.values,
                               names=reason_counts.index,
                               title='주요 거래 결정 이유')
            st.plotly_chart(fig_reasons)

        # 포트폴리오 현황
        st.header('💼 포트폴리오 현황')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latest_btc = df['btc_balance'].iloc[-1]
            btc_value = latest_btc * df['btc_krw_price'].iloc[-1]
            latest_krw = df['krw_balance'].iloc[-1]
            total_value = btc_value + latest_krw
            
            # 자산 분포 파이 차트
            portfolio_data = pd.DataFrame({
                '자산': ['BTC', 'KRW'],
                '금액': [btc_value, latest_krw]
            })
            fig_portfolio = px.pie(portfolio_data, 
                                 values='금액',
                                 names='자산',
                                 title='자산 분포')
            st.plotly_chart(fig_portfolio)
            
        with col2:
            st.metric("총 자산가치", f"{total_value:,.0f} KRW")
            st.metric("BTC 보유량", f"{latest_btc:.8f} BTC")
            st.metric("KRW 보유량", f"{latest_krw:,.0f} KRW")
            
        with col3:
            # 최근 거래 기록
            st.subheader("최근 거래")
            recent_trades = df.tail(3)[['timestamp', 'decision', 'percentage', 'reason']]
            recent_trades['timestamp'] = pd.to_datetime(recent_trades['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent_trades)

        # AI 분석 인사이트
        st.header('🤖 AI 분석 인사이트')
        
        # 현재 시장 상태 분석
        market_status = "중립"
        if latest_rsi > 70:
            market_status = "과매수"
        elif latest_rsi < 30:
            market_status = "과매도"
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("현재 시장 상태")
            st.info(f"""
            - 시장 상태: {market_status}
            - RSI: {latest_rsi:.1f}
            - 24시간 변동률: {price_change:+.2f}%
            - 최근 거래 유형: {df['decision'].iloc[-1]}
            """)
            
        with col2:
            st.subheader("다음 거래 예측")
            next_trade_prediction = "관망" if market_status == "중립" else "매수" if market_status == "과매도" else "매도"
            st.warning(f"""
            - 예상 거래 유형: {next_trade_prediction}
            - 근거: {df['reason'].iloc[-1]}
            - 추천 비율: {df['percentage'].iloc[-1]}%
            """)

        # 기술적 분석 섹션에 추가
        st.header('🔍 기술적 분석')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("피보나치 레트레이스먼트")
            latest_data = df.iloc[-1]
            current_price = latest_data['btc_krw_price']
            
            # 피보나치 레벨 표시
            fibo_levels = {
                '0%': latest_data['fibo_0'],
                '23.6%': latest_data['fibo_236'],
                '38.2%': latest_data['fibo_382'],
                '50%': latest_data['fibo_500'],
                '61.8%': latest_data['fibo_618'],
                '78.6%': latest_data['fibo_786'],
                '100%': latest_data['fibo_1']
            }
            
            # 현재가와 가장 가까운 피보나치 레벨 찾기
            closest_level = min(fibo_levels.items(), key=lambda x: abs(x[1] - current_price))
            
            for level, price in fibo_levels.items():
                st.metric(
                    f"피보나치 {level}", 
                    f"{price:,.0f} KRW",
                    "현재 레벨" if level == closest_level[0] else None
                )

        with col2:
            st.subheader("엘리엇 파동 분석")
            wave_info = {
                'current_wave': latest_data['elliott_wave_count'],
                'direction': latest_data['elliott_wave_direction'],
                'confidence': latest_data['elliott_wave_confidence'],
                'pattern': latest_data['elliott_wave_pattern']
            }
            
            st.info(f"""
            현재 파동: {wave_info['current_wave']}
            방향: {wave_info['direction']}
            신뢰도: {wave_info['confidence']:.1%}
            패턴: {wave_info['pattern']}
            """)

    else:
        st.info("""
        아직 거래 데이터가 없습니다. 다음 단계를 확인해주세요:
        1. autotrade.py가 실행 중인지 확인
        2. 데이터베이스 파일(bitcoin_trades.db)이 생성되었지 확인
        3. 최소 한 번의 거래가 발생했는지 확인
        """)

if __name__ == "__main__":
    main()