import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime
import os

def format_number(value, is_percentage=False):
    """숫자 포맷팅 함수"""
    if pd.isna(value):
        return "0.00%" if is_percentage else "0"
    if is_percentage:
        return f"{value:.2f}%"
    if value >= 1000000:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.8f}"

def get_db_path():
    """데이터베이스 파일 경로 반환"""
    # 현재 스크립트의 디렉토리
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'bitcoin_trades.db')
    
    if not os.path.exists(db_path):
        st.error(f"데이터베이스 파일을 찾을 수 없습니다: {db_path}")
        return None
    return db_path

def calculate_price_change(conn):
    """가격 변동률 계산"""
    try:
        # 최근 두 개의 거래 기록 가져오기
        df = pd.read_sql_query("""
            SELECT btc_krw_price 
            FROM trades 
            ORDER BY timestamp DESC 
            LIMIT 2
        """, conn)
        
        if len(df) >= 2:
            current_price = df['btc_krw_price'].iloc[0]
            previous_price = df['btc_krw_price'].iloc[1]
            return ((current_price - previous_price) / previous_price) * 100
        return 0.00
    except Exception as e:
        st.warning(f"가격 변동률 계산 중 오류: {e}")
        return 0.00

def calculate_profit_rate(conn):
    """전체 수익률 계산"""
    try:
        # 첫 거래와 최근 거래 데이터
        df = pd.read_sql_query("""
            SELECT krw_balance, btc_balance, btc_krw_price
            FROM trades
            ORDER BY timestamp ASC
        """, conn)
        
        if not df.empty:
            initial_total = df['krw_balance'].iloc[0] + (df['btc_balance'].iloc[0] * df['btc_krw_price'].iloc[0])
            current_total = df['krw_balance'].iloc[-1] + (df['btc_balance'].iloc[-1] * df['btc_krw_price'].iloc[-1])
            
            if initial_total > 0:
                return ((current_total - initial_total) / initial_total) * 100
        return 0.00
    except Exception as e:
        st.warning(f"수익률 계산 중 오류: {e}")
        return 0.00

def calculate_win_rate(conn):
    """승률 계산"""
    try:
        df = pd.read_sql_query("""
            SELECT decision, percentage
            FROM trades
            WHERE decision IN ('buy', 'sell')
        """, conn)
        
        if not df.empty:
            total_trades = len(df)
            profitable_trades = len(df[df['percentage'] > 0])
            return (profitable_trades / total_trades) * 100
        return 0.00
    except Exception as e:
        st.warning(f"승률 계산 중 오류: {e}")
        return 0.00

def get_rsi_status(rsi):
    """RSI 상태 판단"""
    if rsi >= 70:
        return "과매수"
    elif rsi <= 30:
        return "과매도"
    else:
        return "중립"

def safe_db_operation(func):
    """데이터베이스 작업 안전성 보장 데코레이터"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except sqlite3.Error as e:
            st.error(f"데이터베이스 오류: {e}")
        except Exception as e:
            st.error(f"예상치 못한 오류: {e}")
        return None
    return wrapper

def get_ai_insight(df):
    """AI 분석 결과 해석"""
    try:
        if df.empty:
            return "대기", "normal", "데이터 없음"
            
        decision = df['decision'].iloc[0]
        reason = df['reason'].iloc[0]
        
        # AI 신호 해석 (delta_color를 Streamlit 허용값으로 변경)
        if decision == 'buy':
            return "매수", "normal", reason  # green -> normal
        elif decision == 'sell':
            return "inverse", "inverse", reason  # red -> inverse
        else:
            return "관망", "off", reason  # gray -> off
            
    except Exception as e:
        st.warning(f"AI 인사이트 생성 중 오류: {e}")
        return "분석 불가", "off", "데이터 오류"

def format_ai_reason(reason):
    """AI 분석 이유를 깔끔하게 포맷팅"""
    if isinstance(reason, list):
        return " | ".join(reason)
    return str(reason)

def get_trend_status(rsi, decision, reason):
    """현재 시장 동향 분석"""
    trend = "중립" if 45 <= rsi <= 55 else "상승" if rsi > 55 else "하락"
    strength = "강" if rsi > 70 or rsi < 30 else "중" if rsi > 60 or rsi < 40 else "약"
    return f"{strength}{trend}"

def main():
    st.title('Bitcoin Trading Dashboard')
    
    try:
        db_path = get_db_path()
        if db_path is None:
            return
            
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("""
            SELECT timestamp, decision, percentage, reason,
                   btc_balance, krw_balance, btc_krw_price, rsi
            FROM trades 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, conn)
        
        if not df.empty:
            # AI 트레이딩 인사이트 섹션
            st.header('🤖 AI 트레이딩 인사이트')
            
            # 현재 시장 상황
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # 주요 지표
                current_price = df['btc_krw_price'].iloc[0]
                price_change = calculate_price_change(conn)
                rsi_value = df['rsi'].iloc[0]
                
                metrics_df = pd.DataFrame({
                    '지표': ['현재가', 'RSI', '변동률'],
                    '값': [
                        f"{format_number(current_price)} KRW",
                        f"{format_number(rsi_value)}",
                        f"{format_number(price_change, True)}"
                    ]
                })
                st.dataframe(metrics_df, hide_index=True)
            
            with col2:
                # AI 분석 결과
                action, color, reason = get_ai_insight(df)
                trend_status = get_trend_status(rsi_value, action, reason)
                
                st.markdown(f"""
                    ### AI 분석 결과
                    - **현재 동향**: {trend_status}
                    - **매매 신호**: {action}
                    - **판단 근거**: {format_ai_reason(reason)}
                """)
            
            # 거래 성과
            st.markdown("---")
            col3, col4 = st.columns(2)
            
            with col3:
                win_rate = calculate_win_rate(conn)
                st.metric(
                    "AI 예측 정확도",
                    format_number(win_rate, True),
                    "최근 10회 거래 기준"
                )
            
            with col4:
                profit_rate = calculate_profit_rate(conn)
                st.metric(
                    "누적 수익률",
                    format_number(profit_rate, True),
                    "초기 투자 대비"
                )
            
            # 최근 AI 분석 히스토리
            st.markdown("### 📊 최근 AI 분석 히스토리")
            trades_df = pd.read_sql_query("""
                SELECT 
                    timestamp,
                    decision,
                    reason,
                    percentage
                FROM trades 
                ORDER BY timestamp DESC 
                LIMIT 3
            """, conn)
            
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            trades_df['decision'] = trades_df['decision'].map({'buy': '매수', 'sell': '매도', 'hold': '관망'})
            
            for _, row in trades_df.iterrows():
                with st.expander(f"{row['timestamp']} - {row['decision']}"):
                    st.write(f"**판단 근거**: {format_ai_reason(row['reason'])}")
                    st.write(f"**수익률**: {row['percentage']}%")

            # 포트폴리오 현황
            st.header('💼 포트폴리오 현황')
            col1, col2 = st.columns(2)
            
            with col1:
                total_krw = df['krw_balance'].iloc[0]
                total_btc_krw = df['btc_balance'].iloc[0] * df['btc_krw_price'].iloc[0]
                
                fig = px.pie(
                    values=[total_krw, total_btc_krw],
                    names=['KRW', 'BTC'],
                    title='자산 분포'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("총 자산가치")
                st.write(f"**{format_number(total_krw + total_btc_krw)} KRW**")
                
                st.write("BTC 보유량")
                st.write(f"**{format_number(df['btc_balance'].iloc[0])} BTC**")
                
                st.write("KRW 보유량")
                st.write(f"**{format_number(df['krw_balance'].iloc[0])} KRW**")

    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
    
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
