import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import json
from openai import OpenAI
import ta
from ta.utils import dropna
import time
import requests
import base64
from PIL import Image
import io
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel
from openai import OpenAI
import sqlite3
import schedule

class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str

class TechnicalDecision(BaseModel):
    decision: str
    percentage: int
    confidence: float
    reason: str

def init_db():
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
                  reflection TEXT)''')
    conn.commit()
    return conn

def log_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection=''):
    try:
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        
        c.execute("""INSERT INTO trades 
                     (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection))
        
        conn.commit()
        logger.info(f"거래 기록 저장 완료: {decision}, {btc_balance} BTC, {krw_balance} KRW")
        
    except Exception as e:
        logger.error(f"거래 기록 저장 중 오류 발생: {e}")
        conn.rollback()

def get_recent_trades(conn, days=7):
    c = conn.cursor()
    seven_days_ago = (datetime.now() - timedelta(days=days)).isoformat()
    c.execute("SELECT * FROM trades WHERE timestamp > ? ORDER BY timestamp DESC", (seven_days_ago,))
    columns = [column[0] for column in c.description]
    return pd.DataFrame.from_records(data=c.fetchall(), columns=columns)

def calculate_performance(trades_df):
    if trades_df.empty:
        return 0
    
    initial_balance = trades_df.iloc[-1]['krw_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_krw_price']
    final_balance = trades_df.iloc[0]['krw_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_krw_price']
    
    return (final_balance - initial_balance) / initial_balance * 100

def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df) # 투자 퍼포먼스 계산
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None
    
    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an AI trading assistant tasked with analyzing recent trading performance and current market conditions to generate insights and improvements for future trading decisions."
            },
            {
                "role": "user",
                "content": f"""
                Recent trading data:
                {trades_df.to_json(orient='records')}
                
                Current market data:
                {current_market_data}
                
                Overall performance in the last 7 days: {performance:.2f}%
                
                Please analyze this data and provide:
                1. A brief reflection on the recent trading decisions
                2. Insights on what worked well and what didn't
                3. Suggestions for improvement in future trading decisions
                4. Any patterns or trends you notice in the market data
                
                Limit your response to 250 words or less.
                """
            }
        ]
    )
    
    return response.choices[0].message.content

def get_db_connection():
    return sqlite3.connect('bitcoin_trades.db')

# 데이터베이스 초기화
init_db()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def head_and_shoulders(df):
    """
    헤드앤숄더 패턴을 감지하는 함수
    """
    try:
        # 최근 20일간의 데이터로 패턴 확인
        recent_data = df[-20:]
        
        # 피크(고점) 찾기
        peaks = []
        for i in range(1, len(recent_data)-1):
            if recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and \
               recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1]:
                peaks.append((i, recent_data['high'].iloc[i]))
        
        if len(peaks) < 3:
            return "No pattern"
        
        # 헤드앤숄더 패턴 조건 확인
        for i in range(len(peaks)-2):
            left_shoulder = peaks[i]
            head = peaks[i+1]
            right_shoulder = peaks[i+2]
            
            # 헤드가 양쪽 숄더보다 높아야 함
            if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
                # 양쪽 숄더의 높이가 비슷해야 함 (20% 오차 허용)
                if abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] < 0.2:
                    return "Head and Shoulders pattern detected"
        
        return "No pattern"
    
    except Exception as e:
        logger.error(f"헤드앤숄더 패턴 감지 중 오류: {e}")
        return "Error in pattern detection"

def fibonacci_retracement(df):
    """
    피보나치 되돌림 레벨을 계산하는 함수
    """
    try:
        # 최근 고점과 저점 찾기
        recent_high = df['high'].max()
        recent_low = df['low'].min()
        
        # 피보나치 되돌림 레벨 계산 (23.6%, 38.2%, 50%, 61.8%, 78.6%)
        diff = recent_high - recent_low
        levels = {
            '0.0': recent_low,
            '0.236': recent_low + 0.236 * diff,
            '0.382': recent_low + 0.382 * diff,
            '0.5': recent_low + 0.5 * diff,
            '0.618': recent_low + 0.618 * diff,
            '0.786': recent_low + 0.786 * diff,
            '1.0': recent_high
        }
        
        return levels
    
    except Exception as e:
        logger.error(f"피보나치 되돌림 계산 중 오류: {e}")
        return None

def add_indicators(df):
    # 기존 지표들 추가 (볼린저 밴드, RSI, MACD 등)
    indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # RSI (Relative Strength Index) 추가
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # MACD (Moving Average Convergence Divergence) 추가
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # 이동평균선 (단기, 장기)
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    
    # 새로 추가된 패턴 탐지 및 피보나치 되돌림
    df['pattern'] = head_and_shoulders(df)
    df['fib_levels'] = pd.Series([fibonacci_retracement(df)] * len(df))
    
    return df

def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['data'][0]
    else:
        logger.error(f"Failed to fetch Fear and Greed Index. Status code: {response.status_code}")
        return None

def analyze_technical_indicators(df_daily, df_hourly, current_price):
    """기술적 지표 기반 기본 분석"""
    try:
        # RSI 기반 과매수/과매도 체크
        daily_rsi = df_daily['rsi'].iloc[-1]
        hourly_rsi = df_hourly['rsi'].iloc[-1]
        
        # 볼린저 밴드 위치 확인
        bb_position = (current_price - df_daily['bb_bbl'].iloc[-1]) / (df_daily['bb_bbh'].iloc[-1] - df_daily['bb_bbl'].iloc[-1])
        
        # MACD 신호
        macd_signal = df_daily['macd_diff'].iloc[-1]
        
        # 이동평균선 크로스 확인
        sma_cross = df_daily['sma_20'].iloc[-1] > df_daily['ema_12'].iloc[-1]
        
        # 신뢰도 및 결정 계산
        confidence = 0.0
        decision = "hold"
        reason = []
        percentage = 0

        # 과매수/과매도 상태 확인
        if daily_rsi > 75 and hourly_rsi > 75:
            decision = "sell"
            confidence += 0.3
            percentage = 50
            reason.append("RSI 과매수 구간")
        elif daily_rsi < 25 and hourly_rsi < 25:
            decision = "buy"
            confidence += 0.3
            percentage = 50
            reason.append("RSI 과매도 구간")

        # 볼린저 밴드 신호
        if bb_position > 0.95:
            if decision == "sell":
                confidence += 0.2
                percentage += 20
            else:
                decision = "sell"
                confidence += 0.2
                percentage = 30
            reason.append("볼린저 밴드 상단 돌파")
        elif bb_position < 0.05:
            if decision == "buy":
                confidence += 0.2
                percentage += 20
            else:
                decision = "buy"
                confidence += 0.2
                percentage = 30
            reason.append("볼린저 밴드 하단 돌파")

        # MACD 신호 확인
        if macd_signal > 0 and decision == "buy":
            confidence += 0.2
            reason.append("MACD 상승 신호")
        elif macd_signal < 0 and decision == "sell":
            confidence += 0.2
            reason.append("MACD 하락 신호")

        return TechnicalDecision(
            decision=decision,
            percentage=min(percentage, 100),
            confidence=min(confidence, 1.0),
            reason=", ".join(reason)
        )

    except Exception as e:
        logger.error(f"기술적 지표 분석 중 오류 발생: {e}")
        return TechnicalDecision(decision="hold", percentage=0, confidence=0.0, reason="분석 오류")

def is_significant_change(df_daily, df_hourly, current_price):
    """중요한 시장 변동 감지"""
    try:
        # 가격 변동성 체크
        daily_volatility = df_daily['close'].pct_change().std()
        hourly_volatility = df_hourly['close'].pct_change().std()
        
        # 거래량 급증 체크
        volume_surge = df_daily['volume'].iloc[-1] > df_daily['volume'].mean() * 1.5
        
        # 패턴 발생 체크
        pattern = head_and_shoulders(df_daily)
        
        return (
            daily_volatility > 0.03 or  # 일간 변동성 3% 이상
            hourly_volatility > 0.02 or  # 시간당 변동성 2% 이상
            volume_surge or  # 거래량 급증
            pattern != "No pattern"  # 특정 패턴 발생
        )
    except Exception as e:
        logger.error(f"변동성 체크 중 오류 발생: {e}")
        return True

def calculate_daily_volatility(df_daily=None):
    """일일 변동성 계산"""
    try:
        if df_daily is None:
            # 최근 30일의 일봉 데이터 가져오기
            df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
            
        if df_daily is None or df_daily.empty:
            logger.error("변동성 계산을 위한 데이터 획득 실패")
            return 0.02  # 기본값 2% 반환
            
        # 일일 변동성 계산 (종가 기준 표준편차)
        daily_returns = df_daily['close'].pct_change().dropna()
        volatility = daily_returns.std()
        
        # 단 값 방지
        volatility = max(min(volatility, 0.1), 0.01)  # 1%~10% 범위로 제한
        
        logger.info(f"계산된 일일 변동성: {volatility:.2%}")
        return volatility
        
    except Exception as e:
        logger.error(f"변동성 계산 중 오류 발생: {e}")
        return 0.02  # 오류 발생시 기본값 2% 반환

def implement_risk_management(upbit, current_position, current_price):
    """위험 관리 로직"""
    try:
        btc_balance = float(current_position.get('btc_balance', 0))
        avg_buy_price = float(current_position.get('avg_buy_price', 0))
        
        if btc_balance > 0 and avg_buy_price > 0:
            profit_rate = (current_price - avg_buy_price) / avg_buy_price * 100
            
            # 동적 손절매/익절매 로직
            daily_volatility = calculate_daily_volatility() * 100  # 퍼센트로 변환
            stop_loss = -3 * daily_volatility  # 변동성의 3배
            take_profit = 5 * daily_volatility  # 변동성의 5배
            
            logger.info(f"현재 수익률: {profit_rate:.2f}%, 손절매 기준: {stop_loss:.2f}%, 익절매 기준: {take_profit:.2f}%")
            
            if profit_rate < stop_loss:
                sell_amount = btc_balance
                if sell_amount * current_price > 5000:
                    order = upbit.sell_market_order("KRW-BTC", sell_amount)
                    logger.info(f"동적 손절매 실행: {profit_rate:.2f}% 손실")
                    return True
            
            elif profit_rate > take_profit:
                sell_amount = btc_balance * 0.5
                if sell_amount * current_price > 5000:
                    order = upbit.sell_market_order("KRW-BTC", sell_amount)
                    logger.info(f"동적 익절매 실행: {profit_rate:.2f}% 수익")
                    return True
        
        return False
        
    except Exception as e:
        logger.error(f"위험 관리 실행 중 오류 발생: {e}")
        return False

def get_ai_analysis(market_data, recent_trades):
    """AI 분석 수행"""
    try:
        if not isinstance(market_data, dict):
            logger.error("Market data is not a dictionary")
            return TradingDecision(decision='hold', percentage=0, reason='Invalid market data format')

        # 매매 타점 데이터에서 technical_data 직접 추출
        technical_data = None
        if '매매 타점 데이터' in market_data:
            technical_data = market_data['매매 타점 데이터'].get('technical_data')
        
        if not technical_data:
            logger.error("Technical data not found in market data")
            logger.info(f"Market Data Keys: {list(market_data.keys())}")
            return TradingDecision(decision='hold', percentage=0, reason='Technical data not available')

        # 분석 요청 데이터 구성
        analysis_request = {
            "market_data": {
                "trend": market_data['trend'],
                "technical_indicators": {
                    "rsi": technical_data['rsi'],
                    "price": {
                        "current": technical_data['current_price'],
                        "bb_upper": technical_data['bb_upper'],
                        "bb_lower": technical_data['bb_lower'],
                        "bb_middle": technical_data['bb_middle']
                    },
                    "volume": {
                        "current": technical_data['volume'],
                        "ma": technical_data['volume_ma']
                    },
                    "patterns": technical_data['patterns']['patterns']
                }
            }
        }
        
        # AI 분석 요청
        prompt = f"""
        Based on the following comprehensive market analysis:
        
        1. Market Context:
        - Current Price: {technical_data['current_price']:,.0f}
        - 24h Volume: {technical_data['volume']:.2f}
        - Volume MA: {technical_data['volume_ma']:.2f}
        
        2. Technical Analysis:
        - Trend: {market_data['trend']['direction']} (Strength: {market_data['trend']['strength']})
        - RSI: {technical_data['rsi']:.2f}
        - Bollinger Bands Position: {(technical_data['current_price'] - technical_data['bb_lower']) / (technical_data['bb_upper'] - technical_data['bb_lower']):.2%}
        
        3. Pattern Analysis:
        - Candlestick Patterns: {', '.join(technical_data['patterns']['patterns'])}
        - Pattern Strength: {technical_data['patterns']['strength']}
        
        4. Recent Market Activity:
        - Trend Reasons: {', '.join(market_data['trend']['reason'])}
        
        Provide a trading decision considering:
        1. Risk management
        2. Current market momentum
        3. Volume analysis
        4. Technical indicator confluence
        
        Response format:
        {
            "decision": "buy/sell/hold",
            "percentage": 0-100,
            "reason": "detailed explanation"
        }
        """
        
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert cryptocurrency trading analyst. Provide trading decisions in JSON format only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )
        
        # 응답 처리
        result_text = response.choices[0].message.content.strip()
        logger.info(f"AI Response: {result_text}")  # 디버깅용 로그 추가
        
        # JSON 파싱 시도
        try:
            # JSON 부분만 추출
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result_json = json.loads(result_text[json_start:json_end])
            else:
                raise ValueError("No JSON found in response")
                
            return TradingDecision(
                decision=result_json.get('decision', 'hold'),
                percentage=result_json.get('percentage', 0),
                reason=result_json.get('reason', 'No reason provided')
            )
        except json.JSONDecodeError as je:
            logger.error(f"JSON 파싱 오류: {je}")
            return TradingDecision(
                decision='hold',
                percentage=0,
                reason=f'JSON parsing error: {str(je)}'
            )
            
    except Exception as e:
        logger.error(f"AI 분석 중 오류 발생: {str(e)}")
        return TradingDecision(
            decision='hold',
            percentage=0,
            reason=f'Error in AI analysis: {str(e)}'
        )

def get_market_data():
    """여러 시간대의 시장 데이터 수집"""
    try:
        # 추세 분석용 데이터
        df_weekly = pyupbit.get_ohlcv("KRW-BTC", interval="week", count=52)  # 주봉 데이터
        if df_weekly is None or df_weekly.empty:
            logger.error("주봉 데이터 수집 실패")
            return None, None, None, None
        df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)    # 일봉 데이터
        if df_daily is None or df_daily.empty:
            logger.error("일봉 데이터 수집 실패")
            return None, None, None, None
        
        # 매매 타점 분석용 데이터
        df_30min = pyupbit.get_ohlcv("KRW-BTC", interval="minute30", count=48)  # 30분봉 이터
        if df_30min is None or df_30min.empty:
            logger.error("30분봉 데이터 수집 실패")
            return None, None, None, None
        df_1min = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=60)    # 1분봉 데이터
        if df_1min is None or df_1min.empty:
            logger.error("1분봉 데이터 수집 실패")
            return None, None, None, None
        
        # 각 데이터프레임에 지표 추가
        for df in [df_weekly, df_daily, df_30min, df_1min]:
            if not df.empty:
                df = add_indicators(df)
        
        return df_weekly, df_daily, df_30min, df_1min
    except Exception as e:
        logger.error(f"시장 데이터 수집 중 오류 발생: {e}")
        return None, None, None, None

def analyze_trend(df_weekly, df_daily):
    """주봉/일봉 기반 추세 분석"""
    try:
        if df_weekly is None or df_daily is None:
            logger.error("추세 분석을 위한 데이터가 부족합니다")
            return {'direction': 'neutral', 'strength': 0, 'reason': ['데이터 부족']}
        
        trend = {
            'direction': 'neutral',
            'strength': 0,
            'reason': []
        }
        
        # 주봉 분석
        weekly_ma20 = df_weekly['sma_20'].iloc[-1]
        weekly_price = df_weekly['close'].iloc[-1]
        weekly_trend = 'bullish' if weekly_price > weekly_ma20 else 'bearish'
        
        # 일봉 분석
        daily_ma20 = df_daily['sma_20'].iloc[-1]
        daily_price = df_daily['close'].iloc[-1]
        daily_trend = 'bullish' if daily_price > daily_ma20 else 'bearish'
        
        # 추세 강도 계산
        if weekly_trend == daily_trend:
            trend['direction'] = weekly_trend
            trend['strength'] = 2
            trend['reason'].append(f"주봉/일봉 모두 {weekly_trend} 추세")
        else:
            trend['direction'] = daily_trend
            trend['strength'] = 1
            trend['reason'].append("단기/장기 추세 불일치")
        
        # 추가적인 추세 확인
        if df_weekly['macd_diff'].iloc[-1] > 0:
            trend['strength'] += 1
            trend['reason'].append("주봉 MACD 상승")
        
        if df_daily['macd_diff'].iloc[-1] > 0:
            trend['strength'] += 1
            trend['reason'].append("일봉 MACD 상승")
            
        # 볼륨 가중 추세 분석
        volume_weighted_price = (df_daily['close'] * df_daily['volume']).sum() / df_daily['volume'].sum()
        if volume_weighted_price > daily_ma20:
            trend['strength'] += 0.5
            trend['reason'].append("거래량 가중 가격이 MA20 상회")
            
        # 추세 지속성 체크
        consecutive_trend = 0
        for i in range(len(df_daily)-1, max(0, len(df_daily)-5), -1):
            if df_daily['close'].iloc[i] > df_daily['close'].iloc[i-1]:
                consecutive_trend += 1
            else:
                break
        if consecutive_trend >= 3:
            trend['strength'] += 0.5
            trend['reason'].append(f"{consecutive_trend}일 연속 상승")
            
        return trend
    
    except Exception as e:
        logger.error(f"추세 분석 중 오류 발생: {e}")
        return {'direction': 'neutral', 'strength': 0, 'reason': ['분석 오류']}

def analyze_elliott_wave(df):
    """엘리엇 파동 분석"""
    try:
        # 최근 데이터 추출 (30개 데이터 포인트)
        recent_data = df[-30:].copy()
        
        # 고점과 저점 기
        peaks = []
        troughs = []
        
        for i in range(1, len(recent_data)-1):
            if recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and \
               recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1]:
                peaks.append((i, recent_data['high'].iloc[i]))
            
            if recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and \
               recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1]:
                troughs.append((i, recent_data['low'].iloc[i]))
        
        # 파동 패턴 분석
        wave_pattern = {
            'current_wave': 0,
            'direction': 'unknown',
            'confidence': 0,
            'reason': []
        }
        
        if len(peaks) >= 3 and len(troughs) >= 2:
            # 상승 파동(1-3-5) 패턴 확인
            if peaks[-1][1] > peaks[-2][1] > peaks[-3][1]:
                wave_pattern['current_wave'] = 5
                wave_pattern['direction'] = 'bullish'
                wave_pattern['confidence'] = 0.7
                wave_pattern['reason'].append("상승 5파동 진행 중")
            
            # 하락 파동(A-C-E) 패턴 확인
            elif peaks[-1][1] < peaks[-2][1] < peaks[-3][1]:
                wave_pattern['current_wave'] = 3
                wave_pattern['direction'] = 'bearish'
                wave_pattern['confidence'] = 0.7
                wave_pattern['reason'].append("하락 3파동 진행 중")
            
            # 조정 파동(2-4) 패턴 확인
            if troughs[-1][1] < troughs[-2][1]:
                wave_pattern['current_wave'] = 4
                wave_pattern['direction'] = 'correction'
                wave_pattern['confidence'] = 0.5
                wave_pattern['reason'].append("조정 4파동 진행 중")
        
        # 피보나치 되돌림과 결합
        fib_levels = fibonacci_retracement(df)
        current_price = df['close'].iloc[-1]
        
        # 파동별 피보나치 레벨 
        if wave_pattern['current_wave'] in [2, 4]:  # 조정 파동
            if current_price > fib_levels['0.618']:
                wave_pattern['confidence'] += 0.2
                wave_pattern['reason'].append("조정 파동 61.8% 되돌림 도달")
        elif wave_pattern['current_wave'] == 3:  # 상승/하락 3파동
            if current_price > fib_levels['1.618']:
                wave_pattern['confidence'] += 0.2
                wave_pattern['reason'].append("3파동 161.8% 확장 도달")
        
        return wave_pattern
    
    except Exception as e:
        logger.error(f"엘리엇 파동 분석 중 오류 발생: {e}")
        return {'current_wave': 0, 'direction': 'unknown', 'confidence': 0, 'reason': ['분석 오류']}

def analyze_entry_point(df_30min, df_1min, trend):
    """30분봉/1분봉 기반 매매 타점 분석"""
    try:
        entry = {
            'action': 'hold',
            'confidence': 0,
            'reason': [],
            'technical_data': {}
        }
        
        # RSI 분석 (기준값 조)
        rsi_30min = df_30min['rsi'].iloc[-1]
        entry['technical_data']['rsi'] = rsi_30min
        if rsi_30min < 35:  # 35로 완화
            entry['action'] = 'buy'
            entry['confidence'] += 0.3
            entry['reason'].append("30분봉 RSI 매매 구간")
        elif rsi_30min > 65:  # 65로 완화
            entry['action'] = 'sell'
            entry['confidence'] += 0.3
            entry['reason'].append("30분봉 RSI 과매수 구간")
        
        # 볼린저밴드 분석 (기준값 조정)
        current_price = df_1min['close'].iloc[-1]
        bb_upper = df_1min['bb_bbh'].iloc[-1]
        bb_lower = df_1min['bb_bbl'].iloc[-1]
        bb_middle = df_1min['bb_bbm'].iloc[-1]
        
        # 추세가 강할 때는 더 적극적인 매매
        if trend['strength'] >= 2:
            if trend['direction'] == 'bullish' and current_price > bb_middle:
                entry['action'] = 'buy'
                entry['confidence'] += 0.3
                entry['reason'].append("상승추세에서 중간밴드 상향 돌파")
            elif trend['direction'] == 'bearish' and current_price < bb_middle:
                entry['action'] = 'sell'
                entry['confidence'] += 0.3
                entry['reason'].append("하락추세에서 중간밴드 상향 돌파")
        
        # 기술적 데이터 추가
        entry['technical_data'].update({
            'current_price': current_price,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_middle': bb_middle,
            'trend_strength': trend['strength'],
            'trend_direction': trend['direction'],
            'volume': df_30min['volume'].iloc[-1],
            'volume_ma': df_30min['volume'].rolling(window=20).mean().iloc[-1],
            'fibo_levels': fibonacci_retracement(df_30min),
            'patterns': detect_candlestick_patterns(df_30min)
        })
        
        logger.info(f"매매 타점 분석 결과: RSI={rsi_30min:.2f}, 현재가격={current_price:,.0f}, BB상단={bb_upper:,.0f}, BB하단={bb_lower:,.0f}, 추세강도={trend['strength']}")
        return entry
        
    except Exception as e:
        logger.error(f"매매 타점 분석 중 오류 발생: {e}")
        return {'action': 'hold', 'confidence': 0, 'reason': ['분석 오류']}

def analyze_volume_profile(df):
    """거래량 프로파일 분석"""
    try:
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        current_volume = df['volume'].iloc[-1]
        
        return {
            'volume_trend': 'high' if current_volume > volume_mean + volume_std else 'low',
            'volume_ratio': current_volume / volume_mean,
            'volume_zscore': (current_volume - volume_mean) / volume_std
        }
    except Exception as e:
        logger.error(f"거래량 프���파일 분석 중 오류: {e}")
        return None

def calculate_risk_score(volatility_data, volume_data):
    """리스크 점수 계산"""
    try:
        # 변동성 점수 (0-1)
        volatility_score = min(volatility_data['natr'] / 10, 1)
        
        # 거래량 점수 (0-1)
        volume_score = min(volume_data['volume_ratio'] / 3, 1)
        
        # 종합 리스크 점수
        risk_score = (volatility_score * 0.7) + (volume_score * 0.3)
        
        return risk_score
    except Exception as e:
        logger.error(f"리스크 점수 계산 중 오류: {e}")
        return 0.5

def optimize_position_size(risk_score, momentum_data):
    """포지션 사이즈 최적화"""
    try:
        # 기본 포지션 사이즈 (리스크 점수 반비례)
        base_size = 1 - (risk_score * 0.5)  # 50-100% 범위
        
        # 모멘텀 조정
        if momentum_data['ppo'] > 0:
            momentum_multiplier = 1.2
        else:
            momentum_multiplier = 0.8
            
        return base_size * momentum_multiplier
    except Exception as e:
        logger.error(f"포지션 사이즈 최적화 중 오류: {e}")
        return 0.5

def analyze_market_conditions(df_daily):
    """시장 상황 분석"""
    try:
        conditions = {}
        
        # RSI 계산
        rsi = ta.momentum.RSIIndicator(df_daily['close']).rsi().iloc[-1]
        conditions['rsi'] = rsi
        
        # MACD 계산
        macd = ta.trend.MACD(df_daily['close'])
        conditions['macd'] = macd.macd().iloc[-1]
        conditions['macd_signal'] = macd.macd_signal().iloc[-1]
        conditions['macd_diff'] = macd.macd_diff().iloc[-1]
        
        # 볼린저 밴드 계산
        bb = ta.volatility.BollingerBands(df_daily['close'])
        conditions['bb_upper'] = bb.bollinger_hband().iloc[-1]
        conditions['bb_lower'] = bb.bollinger_lband().iloc[-1]
        conditions['bb_middle'] = bb.bollinger_mavg().iloc[-1]
        
        # 거래량 분석
        conditions['volume'] = df_daily['volume'].iloc[-1]
        conditions['volume_ma'] = df_daily['volume'].rolling(window=20).mean().iloc[-1]
        
        logger.info(f"시장 상황 분석 완료: RSI={rsi:.2f}")
        return conditions
        
    except Exception as e:
        logger.error(f"시장 상황 분석 중 오류: {e}")
        return None

def enhanced_ai_decision(market_conditions, entry_point):
    """AI 기반 거래 결정"""
    try:
        if market_conditions is None or entry_point is None:
            return {'decision': 'hold', 'position_size': 0, 'reason': '데이터 부족'}

        decision = {
            'decision': 'hold',
            'position_size': 0,
            'reason': []
        }

        # 기술적 데이터 분석
        technical_data = entry_point.get('technical_data', {})
        
        # 피보나치 레벨 분석
        fibo_levels = technical_data.get('fibo_levels', {})
        current_price = technical_data.get('current_price', 0)
        if fibo_levels:
            support = fibo_levels.get('0.618', 0)
            resistance = fibo_levels.get('0.382', 0)
            if current_price < support:
                decision['decision'] = 'buy'
                decision['position_size'] = 0.2
                decision['reason'].append("피보나치 지지선 접근")
            elif current_price > resistance:
                decision['decision'] = 'sell'
                decision['position_size'] = 0.2
                decision['reason'].append("피보나치 저항선 접근")

        # 차트 패턴 분석
        patterns = technical_data.get('patterns', [])
        if patterns:
            bullish_patterns = ['Hammer', 'Morning Star', 'Bullish Engulfing']
            bearish_patterns = ['Shooting Star', 'Evening Star', 'Bearish Engulfing']
            
            for pattern in patterns:
                if pattern in bullish_patterns:
                    decision['decision'] = 'buy'
                    decision['position_size'] = max(decision['position_size'], 0.15)
                    decision['reason'].append(f"강세 패턴 감지: {pattern}")
                elif pattern in bearish_patterns:
                    decision['decision'] = 'sell'
                    decision['position_size'] = max(decision['position_size'], 0.15)
                    decision['reason'].append(f"약세 패턴 감지: {pattern}")

        # 기존 분석 (RSI, 볼린저밴드 등)
        rsi = entry_point.get('rsi', 50)
        if rsi < 30:
            decision['decision'] = 'buy'
            decision['position_size'] = max(decision['position_size'], 0.2)
            decision['reason'].append(f"RSI 과매도({rsi:.1f})")
        elif rsi > 70:
            decision['decision'] = 'sell'
            decision['position_size'] = max(decision['position_size'], 0.2)
            decision['reason'].append(f"RSI 과매수({rsi:.1f})")

        # 추세 강도 반영
        trend_strength = technical_data.get('trend_strength', 0)
        if trend_strength >= 2:
            decision['position_size'] = min(decision['position_size'] * 1.5, 1.0)
            decision['reason'].append(f"강한 추세 감지(강도: {trend_strength})")

        # 최종 결정 정리
        decision['reason'] = ' | '.join(decision['reason']) if decision['reason'] else '특별한 시그널 없음'
        
        logger.info(f"AI 의사결정 결과: {decision}")
        return decision

    except Exception as e:
        logger.error(f"AI 의사결정 중 오류: {e}")
        return {'decision': 'hold', 'position_size': 0, 'reason': f'의사결정 오류: {str(e)}'}

def detect_candlestick_patterns(df):
    """캔들스틱 패턴 감지"""
    try:
        patterns = []
        
        # 최근 3개의 캔들로 패턴 분석
        recent_candles = df.tail(3)
        
        # 망치형(Hammer) 패턴
        last_candle = recent_candles.iloc[-1]
        body = abs(last_candle['open'] - last_candle['close'])
        lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        if lower_shadow > (body * 2) and upper_shadow < body:
            patterns.append("Hammer")
        
        # 역망치형(Inverted Hammer) 패턴
        if upper_shadow > (body * 2) and lower_shadow < body:
            patterns.append("Inverted Hammer")
        
        # 도지(Doji) 패턴
        if body < (last_candle['high'] - last_candle['low']) * 0.1:
            patterns.append("Doji")
        
        return patterns
        
    except Exception as e:
        logger.error(f"캔들스틱 패턴 감지 중 오류: {e}")
        return []

def log_monitoring_data(upbit, trend_data=None, entry_data=None):
    """주기적인 모니터링 데이터 저장"""
    try:
        conn = get_db_connection()
        
        # 현재 잔고 정보 가져오기
        balances = upbit.get_balances()
        btc_balance = float(next((balance['balance'] for balance in balances if balance['currency'] == 'BTC'), 0))
        krw_balance = float(next((balance['balance'] for balance in balances if balance['currency'] == 'KRW'), 0))
        btc_avg_buy_price = float(next((balance['avg_buy_price'] for balance in balances if balance['currency'] == 'BTC'), 0))
        current_price = pyupbit.get_current_price("KRW-BTC")
        
        # 결정 및 이유 구성
        decision = "hold"
        percentage = 0
        reason = "정기 모니터링"
        
        if trend_data and isinstance(trend_data, dict) and entry_data and isinstance(entry_data, dict):
            decision = entry_data.get('action', 'hold')
            percentage = int(entry_data.get('confidence', 0) * 100)
            reasons = []
            if 'reason' in trend_data:
                reasons.extend(trend_data['reason'])
            if 'reason' in entry_data:
                reasons.extend(entry_data['reason'])
            reason = " | ".join(reasons) if reasons else "정기 모니터링"
        
        # DB에 저장
        log_trade(
            conn=conn,
            decision=decision,
            percentage=percentage,
            reason=reason,
            btc_balance=btc_balance,
            krw_balance=krw_balance,
            btc_avg_buy_price=btc_avg_buy_price,
            btc_krw_price=current_price,
            reflection=""
        )
        
        logger.info("모니터링 데이터가 DB에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"모니터링 데이터 저장 중 오류 발생: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def determine_trading_decision(technical_analysis, market_data):
    """기술적 분석과 시장 데이터를 기반으로 거래 결정"""
    try:
        decision = 'hold'
        
        # 추세 확인
        trend = technical_analysis.get('trend', {})
        if trend.get('strength', 0) >= 2:
            decision = 'buy' if trend.get('direction') == 'bullish' else 'sell'
        
        # 모멘텀 확인
        momentum = market_data.get('momentum', {})
        if momentum.get('ppo', 0) > 0 and momentum.get('tsi', 0) > 0:
            decision = 'buy'
        elif momentum.get('ppo', 0) < 0 and momentum.get('tsi', 0) < 0:
            decision = 'sell'
        
        return decision
        
    except Exception as e:
        logger.error(f"거래 결정 도출 중 오류: {e}")
        return 'hold'

def calculate_confidence_score(technical_analysis, market_data):
    """거래 결정에 대한 신뢰도 점수 계산"""
    try:
        confidence = 0.0
        
        # 추세 강도 반영
        trend_strength = technical_analysis.get('trend', {}).get('strength', 0)
        confidence += (trend_strength / 4) * 0.4
        
        # 모멘텀 지표 반영
        momentum = market_data.get('momentum', {})
        if momentum.get('ppo', 0) * momentum.get('tsi', 0) > 0:
            confidence += 0.3
        
        return min(confidence, 1.0)
        
    except Exception as e:
        logger.error(f"신뢰도 점수 계산 �� 오류: {e}")
        return 0.0

def generate_trading_reasoning(technical_analysis, market_data, decision):
    """거래 결정에 대한 상세 근거 생성"""
    try:
        reasons = []
        
        # 추세 분석 근거
        trend = technical_analysis.get('trend', {})
        reasons.append(f"추세 분석: {trend.get('direction', 'neutral')} (강도: {trend.get('strength', 0)})")
        
        # 모멘텀 분석 근거
        momentum = market_data.get('momentum', {})
        reasons.append(f"PPO: {momentum.get('ppo', 0):.2f}, TSI: {momentum.get('tsi', 0):.2f}")
        
        return ' | '.join(reasons)
        
    except Exception as e:
        logger.error(f"거래 근거 생성 중 오류: {e}")
        return "분석 패턴 없음"

def execute_trade(upbit, decision):
    """거래 실행"""
    try:
        if decision['decision'] == "buy":
            # 매수 가능한 KRW 잔고 확인
            krw_balance = upbit.get_balance("KRW")
            if krw_balance is None or float(krw_balance) < 5000:  # 최소 거래금액
                logger.info("매수 가능한 KRW 잔고 부족")
                return None
                
            # 매수할 금액 계산 (confidence에 따라 조정)
            buy_amount = float(krw_balance) * decision['position_size']
            if buy_amount < 5000:
                buy_amount = 5000  # 최소 거래금액
                
            # 매수 주문
            logger.info(f"매수 시도: {buy_amount:,.0f}원")
            result = upbit.buy_market_order("KRW-BTC", buy_amount)
            logger.info(f"매수 결과: {result}")
            
        elif decision['decision'] == "sell":
            # BTC 잔고 확인
            btc_balance = upbit.get_balance("BTC")
            if btc_balance is None or float(btc_balance) <= 0:
                logger.info("매도할 BTC 잔고 없음")
                return None
                
            # 매도할 수량 계산 (confidence에 따라 조정)
            sell_amount = float(btc_balance) * decision['position_size']
            current_price = pyupbit.get_current_price("KRW-BTC")
            
            if current_price is None or sell_amount * current_price < 5000:  # 최소 거래금액 체크
                logger.info("매도 금액이 최소 거래금액 미만")
                return None
                
            # 매도 주문
            logger.info(f"매도 시도: {sell_amount:.8f} BTC")
            result = upbit.sell_market_order("KRW-BTC", sell_amount)
            logger.info(f"매도 결과: {result}")
            
        else:
            logger.info("Hold 상태 유지")
            return None
            
        return result
        
    except Exception as e:
        logger.error(f"거래 실행 중 오류 발생: {e}")
        return None

def run_trading_job():
    """정해진 시간에 실행되는 트레이딩 작업"""
    try:
        logger.info("Trading job 시작...")
        
        # Upbit 객체 생성
        upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
        if not upbit:
            logger.error("Upbit 연결 실패")
            return
            
        # 시장 데이터 수집 및 분석
        df_weekly, df_daily, df_30min, df_1min = get_market_data()
        if any(df is None for df in [df_weekly, df_daily, df_30min, df_1min]):
            logger.error("시장 데이터 수집 실패")
            return
            
        # 시장 상황 분석
        market_conditions = analyze_market_conditions(df_daily)
        
        # 추세 분석
        trend = analyze_trend(df_weekly, df_daily)
        
        # 매매 타점 분석
        entry = analyze_entry_point(df_30min, df_1min, trend)
        
        # AI 분석
        ai_result = enhanced_ai_decision(market_conditions, entry)
        
        # 모니터링 데이터 저장
        log_monitoring_data(upbit, trend, entry)
        
        # 거래 실행 여부 결정
        if ai_result and ai_result['decision'] != 'hold':
            execute_trade(upbit, ai_result)
        
        logger.info("Trading job 완료")
        
    except Exception as e:
        logger.error(f"Trading job 실행 중 오류 발생: {e}")

def setup_schedule():
    # 기존 스케줄
    schedule.every().day.at("08:30").do(run_trading_job)
    schedule.every().day.at("15:30").do(run_trading_job)
    schedule.every().day.at("21:30").do(run_trading_job)
    schedule.every().day.at("23:00").do(run_trading_job)
    schedule.every().day.at("04:30").do(run_trading_job)
    
    # 모니터링 데이터 저장 (5분마다)
    upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
    schedule.every(5).minutes.do(log_monitoring_data, upbit)

def main():
    try:
        # 로깅 설정
        logger.info("자동 거래 프로그램 시작")
        
        # 환경 변수 확인
        if not os.getenv("UPBIT_ACCESS_KEY") or not os.getenv("UPBIT_SECRET_KEY"):
            logger.error("UPBIT API 키가 정되지 않았습니다.")
            raise ValueError("UPBIT API 키 설정이 필요합니다.")
        
        # 즉시 실
        logger.info("초기 거래 작업 실행")
        run_trading_job()
        
        # 스케줄러 실행
        logger.info("스케줄러 시작")
        setup_schedule()
        
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"메인 프로그램 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
