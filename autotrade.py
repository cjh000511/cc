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

# .env 파일에서 환경변수 로드
load_dotenv()
access_key = os.getenv('UPBIT_ACCESS_KEY')
secret_key = os.getenv('UPBIT_SECRET_KEY')

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
                  rsi REAL,
                  macd REAL,
                  volume REAL,
                  bb_upper REAL,
                  bb_lower REAL,
                  bb_middle REAL,
                  fibo_0 REAL,
                  fibo_236 REAL,
                  fibo_382 REAL,
                  fibo_500 REAL,
                  fibo_618 REAL,
                  fibo_786 REAL,
                  fibo_1 REAL,
                  trend_strength INTEGER,
                  trend_direction TEXT,
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
        model="gpt-4",
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

# 데이터베이스 초기���
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
    try:
        if df.empty:
            return {}
            
        recent_high = df['high'].max()
        recent_low = df['low'].min()
        diff = recent_high - recent_low
        
        return {
            '0.0': recent_low,
            '0.236': recent_low + 0.236 * diff,
            '0.382': recent_low + 0.382 * diff,
            '0.5': recent_low + 0.5 * diff,
            '0.618': recent_low + 0.618 * diff,
            '0.786': recent_low + 0.786 * diff,
            '1.0': recent_high
        }
    except Exception as e:
        logger.error(f"피보나치 레트레이스먼트 계산 중 오류: {e}")
        return {}

def add_indicators(df):
    # 기존 지표들 추가 (볼린저 밴드, RSI, MACD 등)
    indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # RSI (Relative Strength Index) 추가
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # MACD (Moving Average Convergence Divergence) ���가
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
            logger.error("일일 변동성 계산을 위한 데이터 획득 실패")
            return 0.02  # 기본값 2% 반환
            
        # 일일 변동성 계산 (종가 기준 표준편차)
        daily_returns = df_daily['close'].pct_change().dropna()
        volatility = daily_returns.std()
        
        # 단 값 지
        volatility = max(min(volatility, 0.1), 0.01)  # 1%~10% 범위로 제한
        
        logger.info(f"계산된 일일 변동성: {volatility:.2%}")
        return volatility
        
    except Exception as e:
        logger.error(f"일일 변동성 계산 중 오류 발생: {e}")
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

def get_market_data(lookback_period=100):
    """충분한 기간의 시장 데이터 가져오기"""
    try:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=lookback_period)
        if df is None or df.empty:
            logging.error("시장 데이터 가져오기 실패")
            return None
        return df
    except Exception as e:
        logging.error(f"시장 데이터 조회 중 오류: {e}")
        return None

def calculate_technical_indicators(df):
    """기술적 지표 계산"""
    try:
        if df is None or df.empty:
            return None
            
        # RSI 계산 (14기간)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD 계산 (12, 26, 9)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # 볼린저 밴드 계산 (20기간, 2표준편차)
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # 이동평균 계산
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # NaN 값 제거
        df = df.dropna()
        
        return df
    except Exception as e:
        logging.error(f"기술적 지표 계산 중 오류: {e}")
        return None

def analyze_market():
    """시장 상황 분석"""
    try:
        df = get_market_data()
        if df is None:
            return None
            
        df = calculate_technical_indicators(df)
        if df is None:
            return None
            
        # 최신 데이터 추출
        latest = df.iloc[-1]
        
        # 차트 패턴 감지
        patterns = detect_chart_patterns(df)
        
        # 엘리엇 파동 분석
        elliott_wave = analyze_elliott_wave(df)
        
        # 피보나치 레벨
        fibo_levels = calculate_fibonacci_levels(df)
        
        # 공포탐욕지수 계산
        fear_greed = calculate_fear_greed_index(df)
        
        market_data = {
            'price': {
                'current': latest['close'],
                'open': latest['open'],
                'high': latest['high'],
                'low': latest['low']
            },
            'indicators': {
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                'macd_diff': latest['macd_diff'],
                'bb_upper': latest['bb_upper'],
                'bb_lower': latest['bb_lower'],
                'bb_middle': latest['bb_middle']
            },
            'volume': {
                'current': latest['volume'],
                'ma': latest['volume_ma']
            },
            'trends': {
                'ma5': latest['ma5'],
                'ma20': latest['ma20']
            },
            'patterns': patterns,
            'elliott_wave': elliott_wave,
            'fibonacci': fibo_levels,
            'fear_greed': fear_greed
        }
        
        logging.info(f"시장 상황 분석 완료: RSI={market_data['indicators']['rsi']:.2f}")
        return market_data
        
    except Exception as e:
        logging.error(f"시장 분석 중 오류: {e}")
        return None

def check_balance_and_conditions(upbit, decision):
    """잔고 확인 및 거래 조건 검증"""
    try:
        krw_balance = float(upbit.get_balance("KRW"))
        btc_balance = float(upbit.get_balance("KRW-BTC"))
        current_price = pyupbit.get_current_price("KRW-BTC")
        
        if decision['decision'] == 'buy':
            if krw_balance < 5000:  # 최소 거래금액
                logging.info("매수 가능한 KRW 잔고 부족")
                return False
        elif decision['decision'] == 'sell':
            if btc_balance <= 0:
                logging.info("매도할 BTC 잔고 없음")
                return False
                
        return True
    except Exception as e:
        logging.error(f"잔고 확인 중 오류: {e}")
        return False

def execute_trading_strategy():
    """트레이딩 전략 실행"""
    try:
        # 시장 분석
        market_data = analyze_market()
        if not market_data:
            logging.error("시장 분석 실패")
            return
            
        # 매매 타점 분석
        technical_data = analyze_entry_point(market_data)
        if not technical_data:
            logging.error("매매 타점 분석 실패")
            return
            
        logging.info(f"매매 타점 분석 결과: RSI={technical_data['rsi']:.2f}, "
                    f"현재가격={technical_data['current_price']:,.0f}, "
                    f"BB상단={technical_data['bb_upper']:,.0f}, "
                    f"BB하단={technical_data['bb_lower']:,.0f}, "
                    f"추세강도={technical_data['trend_strength']}")
        
        # AI 분석
        logging.info("AI 분석 시작...")
        ai_decision = analyze_with_ai(market_data, technical_data)
        if not ai_decision:
            logging.error("AI 분석 실패")
            return
            
        logging.info(f"AI 분석 완료: {ai_decision}")
        
        # 잔고 및 조건 확인
        if not check_balance_and_conditions(upbit, ai_decision):
            return
            
        # 거래 실행
        execute_trade(ai_decision)
        
        # 모니터링 데이터 저장
        save_monitoring_data(market_data, technical_data, ai_decision)
        
    except Exception as e:
        logging.error(f"거래 전략 실행 중 오류: {e}")

def main():
    """메인 함수"""
    logging.basicConfig(level=logging.INFO)
    logging.info("자동 거래 프로그램 시작")
    
    try:
        # 초기 설정
        initialize_database()
        
        # 초기 실행
        logging.info("초기 거래 작업 실행")
        execute_trading_strategy()
        
        # 스케줄러 설정
        logging.info("스케줄러 시작")
        schedule.every().day.at("09:45").do(execute_trading_strategy) 
        schedule.every().day.at("17:00").do(execute_trading_strategy)  
        schedule.every().day.at("23:00").do(execute_trading_strategy)  
        schedule.every().day.at("04:00").do(execute_trading_strategy) 
        
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {e}")

def analyze_with_ai(market_data, technical_data):
    """AI 기반 매매 분석"""
    try:
        prompt = f"""
Based on the following comprehensive market analysis:

1. Price Action:
- Current: {market_data['price']['current']:,.0f} KRW
- Daily Range: {market_data['price']['low']:,.0f} - {market_data['price']['high']:,.0f}

2. Technical Indicators:
- RSI: {market_data['indicators']['rsi']:.2f}
- MACD: {market_data['indicators']['macd']:.2f} (Signal: {market_data['indicators']['macd_signal']:.2f})
- Bollinger Bands:
  * Upper: {market_data['indicators']['bb_upper']:,.0f}
  * Middle: {market_data['indicators']['bb_middle']:,.0f}
  * Lower: {market_data['indicators']['bb_lower']:,.0f}

3. Chart Patterns:
{market_data['patterns']}

4. Elliott Wave Analysis:
- Current Wave: {market_data['elliott_wave']['current_wave']}
- Wave Position: {market_data['elliott_wave']['position']}
- Trend Direction: {market_data['elliott_wave']['trend']}

5. Fibonacci Levels:
{market_data['fibonacci']}

6. Market Sentiment:
- Fear & Greed Index: {market_data['fear_greed']}
- Volume Analysis: {market_data['volume']['current']:.2f} vs MA {market_data['volume']['ma']:.2f}

7. Trend Analysis:
- MA5 vs MA20: {market_data['trends']['ma5']:,.0f} vs {market_data['trends']['ma20']:,.0f}
- Trend Strength: {technical_data['trend_strength']}
- Overall Direction: {technical_data['trend_direction']}

Analyze all these factors and provide a comprehensive trading decision.
Your response must be in valid JSON format with the following structure:
{{
    "decision": "buy/sell/hold",
    "percentage": <number between 0-100>,
    "reason": "<detailed explanation including pattern recognition, wave analysis, and sentiment>",
    "risk_level": "low/medium/high",
    "stop_loss": <suggested stop loss percentage>
}}
"""
        
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert cryptocurrency trading analyst specializing in technical analysis, Elliott Wave Theory, and market psychology. Provide comprehensive trading decisions in strict JSON format only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )
        
        # 응답 처리 및 JSON 파싱
        result_text = response.choices[0].message.content.strip()
        logging.info(f"AI Response: {result_text}")
        
        try:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result_json = json.loads(result_text[json_start:json_end])
            else:
                raise ValueError("No JSON found in response")
                
            return {
                'decision': result_json.get('decision', 'hold'),
                'position_size': result_json.get('percentage', 0) / 100,
                'reason': result_json.get('reason', 'No reason provided'),
                'risk_level': result_json.get('risk_level', 'medium'),
                'stop_loss': result_json.get('stop_loss', 5)
            }
        except json.JSONDecodeError as je:
            logging.error(f"JSON 파싱 오류: {je}")
            return {
                'decision': 'hold',
                'position_size': 0,
                'reason': f'JSON parsing error: {str(je)}'
            }
            
    except Exception as e:
        logging.error(f"AI 분석 중 오류: {e}")
        return {
            'decision': 'hold',
            'position_size': 0,
            'reason': f'Error in AI analysis: {str(e)}'
        }

def initialize_database():
    """데이터베이스 초기화"""
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        c = conn.cursor()
        
        # trades 테이블 생성 (컬럼명 수정)
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      decision TEXT,
                      percentage REAL,
                      reason TEXT,
                      btc_balance REAL,
                      krw_balance REAL,
                      btc_avg_buy_price REAL,
                      btc_krw_price REAL,
                      rsi REAL,
                      macd REAL,
                      volume REAL,
                      bb_upper REAL,
                      bb_middle REAL,
                      bb_lower REAL,
                      trend_strength INTEGER,
                      trend_direction TEXT)''')
                      
        conn.commit()
        conn.close()
        logging.info("데이터베이스 초기화 완료")
        
    except Exception as e:
        logging.error(f"데이터베이스 초기화 중 오류: {e}")

def save_monitoring_data(market_data, technical_data, ai_decision):
    """모니터링 데이터 저장"""
    try:
        conn = sqlite3.connect('bitcoin_trades.db')
        c = conn.cursor()
        
        btc_balance = float(upbit.get_balance("KRW-BTC"))
        krw_balance = float(upbit.get_balance("KRW"))
        avg_buy_price = float(upbit.get_avg_buy_price("KRW-BTC"))
        
        c.execute("""
            INSERT INTO trades (
                timestamp, decision, percentage, reason,
                btc_balance, krw_balance, btc_avg_buy_price,
                btc_krw_price, rsi, macd, volume,
                bb_upper, bb_middle, bb_lower,
                trend_strength, trend_direction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ai_decision['decision'],
            ai_decision['position_size'] * 100,
            ai_decision['reason'],
            btc_balance,
            krw_balance,
            avg_buy_price,
            market_data['price']['current'],
            technical_data['rsi'],
            market_data['indicators']['macd'],
            market_data['volume']['current'],
            market_data['indicators']['bb_upper'],
            market_data['indicators']['bb_middle'],
            market_data['indicators']['bb_lower'],
            technical_data['trend_strength'],
            technical_data['trend_direction']
        ))
        
        conn.commit()
        conn.close()
        logging.info("모니터링 데이터 저장 완료")
        
    except Exception as e:
        logging.error(f"모니터링 데이터 저장 중 오류: {e}")

def analyze_entry_point(market_data):
    """매매 타점 분석"""
    try:
        current_price = market_data['price']['current']
        
        # 시장 데이터 가져오기
        df = get_market_data()
        if df is None:
            return None
        
        # 추세 강도 분석
        trend_strength = 0
        if market_data['trends']['ma5'] > market_data['trends']['ma20']:
            trend_strength += 1
        if market_data['volume']['current'] > market_data['volume']['ma']:
            trend_strength += 1
        if market_data['indicators']['macd'] > market_data['indicators']['macd_signal']:
            trend_strength += 1
            
        # 추세 방향 분석
        trend_direction = 'bullish' if trend_strength >= 2 else 'bearish'
        
        # 피보나치 레벨 계산
        fibo_levels = calculate_fibonacci_levels(df)
        
        technical_data = {
            'rsi': market_data['indicators']['rsi'],
            'current_price': current_price,
            'bb_upper': market_data['indicators']['bb_upper'],
            'bb_lower': market_data['indicators']['bb_lower'],
            'bb_middle': market_data['indicators']['bb_middle'],
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'volume': market_data['volume']['current'],
            'volume_ma': market_data['volume']['ma'],
            'fibo_levels': fibo_levels,
            'patterns': detect_chart_patterns(df)
        }
        
        logging.info(f"매매 타점 분석 결과: RSI={technical_data['rsi']:.2f}, 현재가격={technical_data['current_price']:,.0f}, BB상단={technical_data['bb_upper']:,.0f}, BB하단={technical_data['bb_lower']:,.0f}, 추세강도={technical_data['trend_strength']}")
        return technical_data
        
    except Exception as e:
        logging.error(f"매매 타점 분석 중 오류: {e}")
        return None

def calculate_fibonacci_levels(df):
    """피보나치 레벨 계산"""
    try:
        # 최근 고점과 저점 찾기
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        price_range = recent_high - recent_low
        
        # 피보나치 레벨 계산
        fibo_levels = {
            "0": recent_low,
            "0.236": recent_low + price_range * 0.236,
            "0.382": recent_low + price_range * 0.382,
            "0.5": recent_low + price_range * 0.5,
            "0.618": recent_low + price_range * 0.618,
            "0.786": recent_low + price_range * 0.786,
            "1": recent_high
        }
        
        # 현재가와 가장 가까운 피보나치 레벨 찾기
        current_price = df['close'].iloc[-1]
        closest_level = min(fibo_levels.items(), key=lambda x: abs(float(x[1]) - current_price))
        
        return {
            "levels": fibo_levels,
            "closest_level": closest_level[0],
            "price_at_level": closest_level[1]
        }
        
    except Exception as e:
        logging.error(f"피보나치 레벨 계산 중 오류: {e}")
        return {
            "levels": {},
            "closest_level": "Unknown",
            "price_at_level": 0
        }

def execute_trade(decision):
    """거래 실행"""
    try:
        if decision['decision'] == 'buy':
            # 매수 가능한 KRW 잔고 확인
            krw_balance = float(upbit.get_balance("KRW"))
            if krw_balance < 5000:  # 최소 거래금액
                logging.info("매수 가능한 KRW 잔고 부족")
                return
                
            # 매수 금액 계산 (총 잔고의 30%)
            buy_amount = krw_balance * decision['position_size']
            if buy_amount >= 5000:
                upbit.buy_market_order("KRW-BTC", buy_amount)
                logging.info(f"매수 주문 실행: {buy_amount:,.0f}원")
                
        elif decision['decision'] == 'sell':
            # 매도 가능한 BTC 잔고 확인
            btc_balance = float(upbit.get_balance("KRW-BTC"))
            if btc_balance <= 0:
                logging.info("매도할 BTC 잔고 없음")
                return
                
            # 매도 수량 계산 (보유량의 30%)
            sell_amount = btc_balance * decision['position_size']
            if sell_amount > 0:
                upbit.sell_market_order("KRW-BTC", sell_amount)
                logging.info(f"매도 주문 실행: {sell_amount:.8f}BTC")
                
    except Exception as e:
        logging.error(f"거래 실행 중 오류: {e}")

def setup_schedule():
    """스케줄 설정"""
    try:
        schedule.every().hour.at(":00").do(execute_trading_strategy)
        logging.info("스케줄러 설정 완료")
    except Exception as e:
        logging.error(f"스케줄러 설정 중 오류: {e}")

def detect_chart_patterns(df):
    """차트 패턴 감지"""
    try:
        patterns = []
        
        # 캔들스틱 패턴
        if is_doji(df):
            patterns.append("Doji")
        if is_hammer(df):
            patterns.append("Hammer")
        if is_shooting_star(df):
            patterns.append("Shooting Star")
            
        # 추세 패턴
        if is_double_top(df):
            patterns.append("Double Top")
        if is_double_bottom(df):
            patterns.append("Double Bottom")
        if is_head_and_shoulders(df):
            patterns.append("Head and Shoulders")
            
        return patterns if patterns else ["No significant patterns detected"]
        
    except Exception as e:
        logging.error(f"차트 패턴 감지 중 오류: {e}")
        return ["Pattern detection error"]

def analyze_elliott_wave(df):
    """엘리엇 파동 분석"""
    try:
        # 최근 100개 데이터로 파동 분석
        wave_data = df.tail(100)
        
        # 추세 방향 확인
        trend = "bullish" if wave_data['close'].iloc[-1] > wave_data['close'].iloc[-20] else "bearish"
        
        # 파동 위치 추정
        wave_position = estimate_wave_position(wave_data)
        
        return {
            "current_wave": wave_position['wave'],
            "position": wave_position['position'],
            "trend": trend,
            "confidence": wave_position['confidence']
        }
        
    except Exception as e:
        logging.error(f"엘리엇 파동 분석 중 오류: {e}")
        return {
            "current_wave": "Unknown",
            "position": "Unknown",
            "trend": "Unknown",
            "confidence": 0
        }

def calculate_fibonacci_levels(df):
    """피보나치 레벨 계산"""
    try:
        # 최근 고점과 저점 찾기
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        price_range = recent_high - recent_low
        
        # 피보나치 레벨 계산
        fibo_levels = {
            "0": recent_low,
            "0.236": recent_low + price_range * 0.236,
            "0.382": recent_low + price_range * 0.382,
            "0.5": recent_low + price_range * 0.5,
            "0.618": recent_low + price_range * 0.618,
            "0.786": recent_low + price_range * 0.786,
            "1": recent_high
        }
        
        # 현재가와 가장 가까운 피보나치 레벨 찾기
        current_price = df['close'].iloc[-1]
        closest_level = min(fibo_levels.items(), key=lambda x: abs(float(x[1]) - current_price))
        
        return {
            "levels": fibo_levels,
            "closest_level": closest_level[0],
            "price_at_level": closest_level[1]
        }
        
    except Exception as e:
        logging.error(f"피보나치 레벨 계산 중 오류: {e}")
        return {
            "levels": {},
            "closest_level": "Unknown",
            "price_at_level": 0
        }

def calculate_fear_greed_index(df):
    """공포탐욕지수 계산"""
    try:
        # 변동성 계산
        volatility = df['close'].pct_change().std() * 100
        
        # 거래량 변화
        volume_change = (df['volume'].iloc[-1] / df['volume'].mean() - 1) * 100
        
        # RSI 기반 과매수/과매도
        rsi = df['rsi'].iloc[-1]
        
        # 가격 모멘텀
        price_momentum = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
        
        # 지표들을 종합하여 0-100 사이의 지수로 변환
        fear_greed = calculate_composite_index(volatility, volume_change, rsi, price_momentum)
        
        return {
            "index": fear_greed,
            "status": get_market_sentiment(fear_greed),
            "components": {
                "volatility": volatility,
                "volume_change": volume_change,
                "rsi": rsi,
                "price_momentum": price_momentum
            }
        }
        
    except Exception as e:
        logging.error(f"공포탐욕지수 계산 중 오류: {e}")
        return {
            "index": 50,
            "status": "Neutral",
            "components": {}
        }

# 보조 함수들
def is_doji(df):
    """도지 캔들 패턴 감지"""
    latest = df.iloc[-1]
    return abs(latest['open'] - latest['close']) <= (latest['high'] - latest['low']) * 0.1

def is_hammer(df):
    """해머 캔들 패턴 감지"""
    latest = df.iloc[-1]
    body = abs(latest['open'] - latest['close'])
    lower_shadow = min(latest['open'], latest['close']) - latest['low']
    return lower_shadow > body * 2

def is_shooting_star(df):
    """슈팅스타 캔들 패턴 감지"""
    latest = df.iloc[-1]
    body = abs(latest['open'] - latest['close'])
    upper_shadow = latest['high'] - max(latest['open'], latest['close'])
    return upper_shadow > body * 2

def estimate_wave_position(df):
    """엘리엇 파동 위치 추정"""
    try:
        # 간단한 파동 위치 추정 로직
        price_changes = df['close'].pct_change()
        recent_changes = price_changes.tail(5)
        
        if recent_changes.mean() > 0:
            if recent_changes.iloc[-1] > 0:
                return {"wave": "Wave 3", "position": "Middle", "confidence": 0.7}
            else:
                return {"wave": "Wave 4", "position": "Correction", "confidence": 0.6}
        else:
            if recent_changes.iloc[-1] < 0:
                return {"wave": "Wave 5", "position": "End", "confidence": 0.5}
            else:
                return {"wave": "Wave 2", "position": "Correction", "confidence": 0.6}
                
    except Exception:
        return {"wave": "Unknown", "position": "Unknown", "confidence": 0}

def calculate_composite_index(volatility, volume_change, rsi, momentum):
    """공포탐욕지수 종합 계산"""
    try:
        # 각 지표를 0-100 사이로 정규화
        vol_score = min(max(50 - volatility, 0), 100)
        vol_change_score = min(max(50 + volume_change, 0), 100)
        rsi_score = rsi
        momentum_score = min(max(50 + momentum, 0), 100)
        
        # 가중 평균 계산
        composite = (vol_score * 0.25 + vol_change_score * 0.25 + 
                    rsi_score * 0.25 + momentum_score * 0.25)
                    
        return round(composite, 2)
        
    except Exception:
        return 50

def get_market_sentiment(index):
    """공포탐욕지수 해석"""
    if index >= 80:
        return "Extreme Greed"
    elif index >= 60:
        return "Greed"
    elif index >= 40:
        return "Neutral"
    elif index >= 20:
        return "Fear"
    else:
        return "Extreme Fear"

def is_double_top(df):
    """더블 탑 패턴 감지"""
    try:
        # 최근 20개 봉 데이터로 분석
        recent_data = df.tail(20)
        highs = recent_data['high'].values
        
        # 고점 찾기
        peaks = []
        for i in range(1, len(highs)-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
                
        # 두 개의 비슷한 고점이 있는지 확인
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            price_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1])
            price_threshold = last_two_peaks[0][1] * 0.02  # 2% 오차 허용
            
            if price_diff <= price_threshold:
                return True
                
        return False
        
    except Exception as e:
        logging.error(f"더블 탑 패턴 감지 중 오류: {e}")
        return False

def is_double_bottom(df):
    """더블 바텀 패턴 감지"""
    try:
        # 최근 20개 봉 데이터로 분석
        recent_data = df.tail(20)
        lows = recent_data['low'].values
        
        # 저점 찾기
        troughs = []
        for i in range(1, len(lows)-1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
                
        # 두 개의 비슷한 저점이 있는지 확인
        if len(troughs) >= 2:
            last_two_troughs = troughs[-2:]
            price_diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1])
            price_threshold = last_two_troughs[0][1] * 0.02  # 2% 오차 허용
            
            if price_diff <= price_threshold:
                return True
                
        return False
        
    except Exception as e:
        logging.error(f"더블 바텀 패턴 감지 중 오류: {e}")
        return False

def is_head_and_shoulders(df):
    """헤드앤숄더 패턴 감지"""
    try:
        # 최근 30개 봉 데이터로 분석
        recent_data = df.tail(30)
        highs = recent_data['high'].values
        
        # 고점 찾기
        peaks = []
        for i in range(1, len(highs)-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
                
        # 최소 3개의 고점이 필요
        if len(peaks) >= 3:
            last_three_peaks = peaks[-3:]
            
            # 중간 고점(head)이 양쪽 고점(shoulders)보다 높은지 확인
            left_shoulder = last_three_peaks[0][1]
            head = last_three_peaks[1][1]
            right_shoulder = last_three_peaks[2][1]
            
            # 양쪽 어깨의 높이가 비슷한지 확인 (10% 오차 허용)
            shoulder_diff = abs(left_shoulder - right_shoulder)
            shoulder_threshold = left_shoulder * 0.1
            
            if (head > left_shoulder and 
                head > right_shoulder and 
                shoulder_diff <= shoulder_threshold):
                return True
                
        return False
        
    except Exception as e:
        logging.error(f"헤드앤숄더 패턴 감지 중 오류: {e}")
        return False

# Upbit 객체 초기화
upbit = pyupbit.Upbit(access_key, secret_key)

if __name__ == "__main__":
    main()
