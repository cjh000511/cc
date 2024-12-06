import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import numpy as np
import json
import ta
from ta.utils import dropna
import time
import requests
import base64
from PIL import Image
import io
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, WebDriverException, NoSuchElementException
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel
import sqlite3
import schedule
from tele import send_trading_notification
from ml_model import EnhancedBitcoinPredictor
from google.cloud import translate_v2 as translate
from trading_conditions import TradingConditions
from monitoring import TradingMonitor
import re
import pytz
from risk_management import RiskManager, RiskParameters
from logging.handlers import RotatingFileHandler
from collections import defaultdict
import functools
from functools import lru_cache
from typing import Optional
import threading
from database_manager import DatabaseManager

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str
    confidence: float

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
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("""INSERT INTO trades 
                 (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection))
    conn.commit()

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

def translate_text(text, target_language='ko'):
    """
    Google Cloud Translation API를 사용하여 텍스트를 번역하는 함수
    """
    try:
        # Translation 클라이언트 초기화
        client = translate.Client()
        
        # 번역 수행
        result = client.translate(
            text,
            target_language=target_language
        )
        
        return result['translatedText']
    except Exception as e:
        logger.error(f"번역  오 : {e}")
        return text  # 오류 발생시 원본 텍스트 반

def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)
    
    # 간단한 성능 약으 대체
    reflection = f"""
    최근 7일 거래 성과:
    - 수익률: {performance:.2f}%
    - 거래 횟수: {len(trades_df)}
    - 평균 거래 크기: {trades_df['percentage'].mean():.1f}%
    """
    
    # 영어로 된 응답을 한국어로 번역
    translated_reflection = translate_text(reflection)
    return translated_reflection

def get_db_connection():
    return sqlite3.connect('bitcoin_trades.db')

# 데이터베이스 초기화
init_db()

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
        
        # 헤앤숄더 패 조 인
        for i in range(len(peaks)-2):
            left_shoulder = peaks[i]
            head = peaks[i+1]
            right_shoulder = peaks[i+2]
            
            # 헤드가 양쪽 숄더보다 높아야 함
            if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
                #  숄더의 높이가 비슷해야 함 (20% 오차 허용)
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

def add_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    기술적 지표 계산 함수
    """
    try:
        # 데이터프레임 복사하여 원본 보존
        df = df.copy()
        
        # 볼린저 밴드
        bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
        df['bb_bbm'] = bb.bollinger_mavg()
        df['bb_bbh'] = bb.bollinger_hband()
        df['bb_bbl'] = bb.bollinger_lband()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        # 이동평균선
        df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['sma_200'] = ta.trend.SMAIndicator(close=df['close'], window=200).sma_indicator()

        # OBV (On-Balance Volume)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

        # CCI (Commodity Channel Index)
        df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()

        # MFI (Money Flow Index)
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()

        # NaN 값 처리
        df = df.ffill().bfill()
        
        return df
        
    except Exception as e:
        logger.error(f"지표 계산 중 오류: {str(e)}")
        return None

def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['data'][0]
    else:
        logger.error(f"Failed to fetch Fear and Greed Index. Status code: {response.status_code}")
        return None

def setup_chrome_options():
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--headless=new")  # 새로운 헤드리스 모드 사용
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-software-rasterizer")  # 추가
    chrome_options.add_argument("--disable-webgl")  # 추가
    chrome_options.add_argument("--ignore-certificate-errors")  # 추가
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_argument("--log-level=3")  # 로그 레벨 최소화
    return chrome_options

def create_driver():
    logger.info("ChromeDriver 정 중...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=setup_chrome_options())
    return driver

def click_element_by_xpath(driver, xpath, element_name, wait_time=10):
    try:
        element = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        # 요소가 뷰포트에 보일 때까지 스크롤
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        # 요소가 클릭 가능할 때까지 대기
        element = WebDriverWait(driver, wait_time).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        element.click()
        logger.info(f"{element_name} 클릭 완료")
        time.sleep(2)  # 클릭  잠시 대기
    except TimeoutException:
        logger.error(f"{element_name} 요소를 찾는 데 시간이 초과되었습니다.")
    except ElementClickInterceptedException:
        logger.error(f"{element_name} 요소를 클릭할 수 없습니다. 다른 요소에 가려져 있을 수 있습니다.")
    except NoSuchElementException:
        logger.error(f"{element_name} 요소를 찾을 수 습니다.")
    except Exception as e:
        logger.error(f"{element_name} 클릭 중 류 발생: {e}")

def perform_chart_actions(driver):
    # 시  메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]",
        "시간 메뉴"
    )
    
    # 1시간 옵션 택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]",
        "1시간 옵션"
    )
    
    # 지표 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    
    # 볼린저 밴드 옵션 택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[15]",
        "볼린 밴드 옵션"
    )

def capture_and_encode_screenshot(driver):
    try:
        # 스크린샷 캡처
        png = driver.get_screenshot_as_png()
        
        # PIL Image로 변환
        img = Image.open(io.BytesIO(png))
        
        # 이미지 리사이즈 (OpenAI API 제한에 맞춤)
        img.thumbnail((2000, 2000))
        
        # 이미지를 바이트로 변환
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        
        # base64로 인
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return base64_image
    except Exception as e:
        logger.error(f"스크린샷 캡처 및 인코딩 중 오류 생: {e}")
        return None

def calculate_net_flow(transactions):
    """순 유입량 계산"""
    try:
        inflow = sum(tx['value'] for tx in transactions.get('received', []))
        outflow = sum(tx['value'] for tx in transactions.get('sent', []))
        return inflow - outflow
    except Exception as e:
        logger.error(f"순유입량 계산 중 오류: {e}")
        return 0

def calculate_hashrate_change(analytics):
    """해시레이트 변화율 계산"""
    try:
        current_hashrate = analytics.get('current_hashrate', 0)
        previous_hashrate = analytics.get('previous_hashrate', 0)
        if previous_hashrate > 0:
            return (current_hashrate - previous_hashrate) / previous_hashrate
        return 0
    except Exception as e:
        logger.error(f"해시레이트 변화율 계산 중 오류: {e}")
        return 0

def calculate_transaction_volume(transactions):
    """거래량 계산"""
    try:
        return sum(tx['value'] for tx in transactions.get('all', []))
    except Exception as e:
        logger.error(f"거래량 계산 중 오류: {e}")
        return 0

def calculate_sopr(transactions):
    """SOPR 계산"""
    try:
        if not transactions.get('all'):
            return 1.0
        spent_value = sum(tx['value'] for tx in transactions.get('sent', []))
        created_value = sum(tx['value'] for tx in transactions.get('received', []))
        if created_value > 0:
            return spent_value / created_value
        return 1.0
    except Exception as e:
        logger.error(f"SOPR 계산  오류: {e}")
        return 1.0

def calculate_miner_revenue(dune_data):
    """채굴자 수익 추정"""
    try:
        # 해시레이트와 비트코인 가격을 기반으로 수익 추정
        current_price = pyupbit.get_current_price("KRW-BTC")
        if current_price and dune_data.get('hashrate_change'):
            # 단 익 추정 식
            return current_price * (1 + dune_data['hashrate_change'])
        return 0
    except Exception as e:
        logger.error(f"채굴자 수익 추정 중 오류: {e}")
        return 0

def estimate_mining_cost(dune_data):
    """채굴 비용 추정"""
    try:
        # 해시레이트 기반 비용 추정
        if dune_data.get('hashrate_change'):
            base_cost = 10000  # 기본 채 비 (USD)
            return base_cost * (1 + abs(dune_data['hashrate_change']))
        return 0
    except Exception as e:
        logger.error(f"채굴 비용 추정 중 오류: {e}")
        return 0

def calculate_onchain_score(onchain_data):
    if not onchain_data:
        return 0.0
        
    try:
        scores = {
            'exchange_flow': 0,
            'miner_activity': 0,
            'network_health': 0,
            'market_sentiment': 0
        }
        
        # 거래소 유입/출 수
        if onchain_data['exchange_net_flow'] < 0:  # 거래소에서 빠져나가는 경우
            scores['exchange_flow'] = min(-onchain_data['exchange_net_flow'] / 1000, 1)
        else:
            scores['exchange_flow'] = max(-onchain_data['exchange_net_flow'] / 1000, -1)
            
        # 채굴자 활동 수
        miner_profit_ratio = (onchain_data['miner_revenue'] - onchain_data['miner_expense']) / onchain_data['miner_revenue']
        scores['miner_activity'] = np.tanh(miner_profit_ratio)
        
        # 네트워크 건전성 점수
        scores['network_health'] = np.tanh(onchain_data['hashrate_change'])
        
        # 시장 심리 수 (SOPR 기반)
        scores['market_sentiment'] = np.tanh(onchain_data['sopr'] - 1)
        
        # 가중치 적용
        weights = {
            'exchange_flow': 0.3,
            'miner_activity': 0.2,
            'network_health': 0.2,
            'market_sentiment': 0.3
        }
        
        final_score = sum(score * weights[key] for key, score in scores.items())
        return final_score
        
    except Exception as e:
        logger.error(f"온체인 점수 계산 중 오류: {e}")
        return 0.0

def analyze_timeframe(df):
    """각 시간대별 기술적 지표 분석"""
    try:
        signals = {}
        
        # RSI 분석
        rsi = calculate_rsi(df['close'])
        signals['rsi'] = '과매도' if rsi.iloc[-1] < 30 else '과매수' if rsi.iloc[-1] > 70 else '중립'
        
        # MACD 분석
        macd_score = calculate_macd_score(df)
        signals['macd'] = '상승세' if macd_score > 0.6 else '하락추세' if macd_score < 0.4 else '중립'
        
        # 이동평균선 분석
        ma20 = df['close'].rolling(window=20).mean()
        current_price = df['close'].iloc[-1]
        signals['ma'] = '상향돌파' if current_price > ma20.iloc[-1] else '하향돌파'
        
        return signals
        
    except Exception as e:
        logger.error(f"시간대별 분석 중 오류: {e}")
        return {'error': str(e)}

def get_ai_trading_decision(df_daily, predicted_price, current_price, time_frame_signals, upbit):
    try:
        reason = "거래 분석 결과:\n"  # reason 변수 초기화
        
        # 기술적 지표 분석
        technical_signals = analyze_technical_indicators(df_daily)
        reason += f"기술적 지표:\n{technical_signals}\n"
        
        # ML 예측 가중치 계산
        ml_weight = calculate_ml_prediction_weight(predicted_price, current_price)
        reason += f"ML 예측 가중치: {ml_weight:.2f}\n"
        
        # 최종 결정
        decision = combine_signals(technical_signals, ml_weight, time_frame_signals)
        
        return TradingDecision(
            decision=decision['action'],
            percentage=decision['percentage'],
            reason=reason,
            confidence=decision['confidence']
        )
    except Exception as e:
        logger.error(f"거래 결정 중 오류: {e}")
        return None

def retry_on_failure(max_retries=3, delay=1):
    """개선된 재시도 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            # 에러 유형별 재시도 설정
            retry_errors = {
                'ConnectionError': {'max_retries': 5, 'delay': 2},
                'TimeoutError': {'max_retries': 4, 'delay': 3},
                'RequestException': {'max_retries': 3, 'delay': 1},
            }
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_type = e.__class__.__name__
                    
                    # 에러 유형별 처리
                    retry_config = retry_errors.get(error_type, {'max_retries': max_retries, 'delay': delay})
                    if attempt >= retry_config['max_retries'] - 1:
                        logger.error(f"{func.__name__} 최종 실패: {e}")
                        break
                        
                    # 지수 백오프 적용
                    wait_time = retry_config['delay'] * (2 ** attempt)
                    logger.warning(f"{func.__name__} 재시도 ({attempt + 1}/{retry_config['max_retries']}): {e}")
                    logger.info(f"대기 시간: {wait_time}초")
                    time.sleep(wait_time)
            
            # 실패 처리
            if func.__name__ == 'execute_trade':
                send_trading_notification(
                    "거래 실패",
                    0,
                    f"거래 실행 중 오류: {str(last_exception)}\n"
                    f"에러 유형: {error_type}\n"
                    f"시도 횟수: {attempt + 1}"
                )
            return None
            
        return wrapper
    return decorator

@retry_on_failure(max_retries=3)
def execute_trade(result, upbit, current_price, btc_balance, krw_balance):
    """거래 실행 함수"""
    try:
        # 거래 실행 전 모니터링 시스템 체크
        monitor = TradingMonitor()  # EnhancedMonitor에서 TradingMonitor로 변경
        market_status = monitor.check_market_conditions(
            current_price=current_price,
            volume=get_current_volume(),
            profit=calculate_current_profit(upbit)
        )
        
        if market_status:
            logger.warning(f"위험 감지: {market_status}")
            return None
            
        # 래 실행
        if result.decision == "buy":
            order_result = upbit.buy_market_order(
                "KRW-BTC", 
                krw_balance * (result.percentage / 100)
            )
        else:
            order_result = upbit.sell_market_order(
                "KRW-BTC", 
                btc_balance * (result.percentage / 100)
            )
            
        # 거래 성공 시 로깅 및 알림
        if order_result:
            logger.info(f"거래 성공: {result.decision}, {result.percentage}%")
            # 텔레그램 알림 전송 시도
            try:
                send_trading_notification(
                    result.decision,
                    result.percentage,
                    f"거래 성공\n결정: {result.decision}\n비율: {result.percentage}%\n이유: {result.reason}"
                )
                logger.info("텔레그램 알림 전송 완료")
            except Exception as e:
                logger.error(f"텔레그램 알림 전송 실패: {e}")
            
        return order_result
        
    except Exception as e:
        logger.error(f"거래 실행 중 오류: {e}")
        return None

@retry_on_failure(max_retries=2)
def get_current_volume():
    """현재 거래량 조회"""
    try:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=1)
        if df is not None:
            return df['volume'].iloc[-1]
        return 0
    except Exception as e:
        logger.error(f"거래량 조회 중 오류: {e}")
        return 0

def get_current_volatility(upbit=None):
    """현재 변동성 계산"""
    try:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
        if df is not None:
            returns = df['close'].pct_change()
            return returns.std()
        return 0.02  # 기본값
    except Exception as e:
        logger.error(f"변동성 계산 중 오류: {e}")
        return 0.02

def calculate_rsi(prices, period=14):
    """RSI(Relative Strength Index) 계"""
    try:
        # 가 변화 계산
        delta = prices.diff()
            
        # 상승/하락 분
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        # 평균 계산
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # RS 계산
        rs = avg_gain / avg_loss
        
        # RSI 계산
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logger.error(f"RSI 계산 중 오류: {e}")
        return None

def calculate_macd(prices, slow=26, fast=12, signal=9):
    """MACD(Moving Average Convergence Divergence) 계산"""
    try:
        # 지수이동평균(EMA) 계산
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        
        # MACD 라인 계산
        macd = exp1 - exp2
        
        # 시그널 라인 계산
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return macd, signal_line
        
    except Exception as e:
        logger.error(f"MACD 계산 중 오류: {e}")
        return None, None

def calculate_bollinger(prices, window=20, num_std=2):
    try:
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return rolling_mean, upper_band, lower_band
    except Exception as e:
        logger.error(f"볼린저 밴드 계산 중 오류: {e}")
        return None, None, None

def get_average_buy_price(upbit):
    """평균 매수 조회"""
    try:
        balances = upbit.get_balances()
        btc_balance = next((balance for balance in balances if balance['currency'] == 'BTC'), None)
        
        if btc_balance:
            return float(btc_balance['avg_buy_price'])
        return 0
        
    except Exception as e:
        logger.error(f"평균 매수가 조회 중 오류: {e}")
        return 0

def calculate_current_profit(upbit):
    """현재 수익률 계산"""
    try:
        balances = upbit.get_balances()
        btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
        avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'BTC'), 0)
        
        if btc_balance == 0 or avg_buy_price == 0:
            return 0.0
            
        current_price = pyupbit.get_current_price("KRW-BTC")
        if current_price is None:
            return 0.0
            
        return (current_price - avg_buy_price) / avg_buy_price
        
    except Exception as e:
        logger.error(f"수익률 계산 중 오류: {e}")
        return 0.0

def calculate_dynamic_position_size(current_price, krw_balance, btc_balance, confidence, upbit):
    """동적 포지션 사이징 계산"""
    try:
        # 기본 지션  (전체 자산의 1~5%)
        base_position = 0.01  # 1%
        
        # 신뢰도에 따른 지 조정
        confidence_multiplier = min(confidence * 2, 1.5)  # 최대 1.5배
        
        # 변동성 따른 조정
        volatility = get_current_volatility(upbit)
        volatility_multiplier = max(0.5, 1 - volatility)  # 변동성이 높을수록 작은 포지
        
        # 최종 포지션 크기 계산
        final_position_size = krw_balance * base_position * confidence_multiplier * volatility_multiplier
        
        # 최소 거래금액 확
        return max(final_position_size, 5000)
        
    except Exception as e:
        logger.error(f"포지션 사이징  중 오류: {e}")
        return 5000  # 오류 시 최소 금액

def set_stop_orders(upbit, market, stop_loss_price, take_profit_price):
    """손/익절 주문 등록"""
    try:
        # 현 보유 수량 확인
        balances = upbit.get_balances()
        btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
        
        if btc_balance <= 0:
            return None
            
        # 손절 주문
        stop_loss_order = upbit.sell_limit_order(
            market,
            stop_loss_price,
            btc_balance
        )
        logger.info(f"손절 주문 등록: {stop_loss_price:,}원")
        
        # 익절 주문
        take_profit_order = upbit.sell_limit_order(
            market,
            take_profit_price,
            btc_balance
        )
        logger.info(f"익절 주문  : {take_profit_price:,}원")
        
        return {
            'stop_loss': stop_loss_order,
            'take_profit': take_profit_order
        }
        
    except Exception as e:
        logger.error(f"손절/익절 주문 등록 중 오류: {e}")
        return None

def get_cached_decision(cache_key):
    """캐시된 거 결정 회"""
    try:
        conn = sqlite3.connect('trading_cache.db')
        cursor = conn.cursor()
        
        # 테이블 성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_cache
            (cache_key TEXT PRIMARY KEY, 
             decision TEXT, 
             percentage INTEGER,
             reason TEXT,
             confidence FLOAT,
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
        ''')
        
        # 1시간 내의 캐시된 결정만 조회
        cursor.execute('''
            SELECT decision, percentage, reason, confidence 
            FROM decision_cache 
            WHERE cache_key = ? 
            AND timestamp > datetime('now', '-1 hour')
        ''', (cache_key,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return TradingDecision(
                decision=result[0],
                percentage=result[1],
                reason=result[2],
                confidence=result[3]
            )
        return None
        
    except Exception as e:
        logger.error(f"캐시 결정 오류: {e}")
        return None

def calculate_rsi_score(df):
    """RSI 기반 점수 계산 (-1 ~ 1)"""
    try:
        rsi = df['rsi'].iloc[-1]  # 지막 RSI 값
        
        # RSI 70 이상은 과매수, 30 이는 과매도
        if rsi >= 70:
            return -1 * (rsi - 70) / 30  # 70~100 구간을 -0~-1로 매핑
        elif rsi <= 30:
            return (30 - rsi) / 30  # 0~30 구간을 1~0으 매핑
        else:
            return (50 - rsi) / 20  # 30~70 구간을 0.5~-0.5로 매핑
            
    except Exception as e:
        logger.error(f"RSI 점수 계산 중 오류: {e}")
        return 0

def calculate_macd_score(df):
    """MACD 기반 점수 계산 (-1 ~ 1)"""
    try:
        if 'macd' not in df.columns or 'macd_signal' not in df.columns:
            return 0
            
        macd = df['macd'].iloc[-1]
        signal = df['macd_signal'].iloc[-1]
        
        # MACD 그널 교차 강도 
        signal_strength = (macd - signal) / abs(signal) if signal != 0 else 0
        
        # 최종 점수 산
        score = np.tanh(signal_strength)
        return score
        
    except Exception as e:
        logger.error(f"MACD 점수 계산 중 오류: {e}")
        return 0

def calculate_bollinger_score(df):
    """볼린저 밴드 기반 점수 계산 (-1 ~ 1)"""
    try:
        current_price = df['close'].iloc[-1]
        upper_band = df['bb_bbh'].iloc[-1]
        lower_band = df['bb_bbl'].iloc[-1]
        middle_band = df['bb_bbm'].iloc[-1]
        
        # 밴드 상 치 계산
        band_width = upper_band - lower_band
        if band_width == 0:
            return 0
            
        relative_position = (current_price - middle_band) / (band_width / 2)
        
        # 점��� 규화 (-1 ~ 1)
        return np.clip(relative_position, -1, 1)
        
    except Exception as e:
        logger.error(f"볼린저 밴드 점수 계산 중 오류: {e}")
        return 0

def calculate_volume_score(df):
    """거래량 기반 점수 계산 (-1 ~ 1)"""
    try:
        # 최근 5개 거래량 평균과 20개 래량 평균 비교
        volume_ma5 = df['volume'].rolling(5).mean().iloc[-1]
        volume_ma20 = df['volume'].rolling(20).mean().iloc[-1]
        
        # 가 변
        price_change = df['close'].pct_change().iloc[-1]
        
        # 거래량 율 계산
        volume_ratio = (volume_ma5 / volume_ma20) - 1
        
        # 거래량 가격 방��성 결합
        score = np.tanh(volume_ratio) * np.sign(price_change)
        return score
        
    except Exception as e:
        logger.error(f"거래량 점수 계산 중 오류: {e}")
        return 0

class RiskManager:
    def __init__(self):
        self.max_daily_loss = -0.05  # 일일 최대 손실 한도
        self.max_position_size = 0.2  # 최대 포지�� 크기
        self.max_consecutive_losses = 3  # 최대 연속 손실 횟수
        self.volatility_threshold = 0.03  # 변동성 임값
        
    def check_risk_levels(self, current_position, daily_pnl, consecutive_losses):
        risk_alerts = []
        
        # 일일 손실 한도 체크
        if daily_pnl < self.max_daily_loss:
            risk_alerts.append(f"일일 손실 한도 초과: {daily_pnl:.2%}")
            
        # 포지션 크기 체크
        if current_position > self.max_position_size:
            risk_alerts.append(f"최대 포지션 크기 초과: {current_position:.2%}")
            
        # 연속 손실 체크
        if consecutive_losses >= self.max_consecutive_losses:
            risk_alerts.append(f"연속 손실 발생: {consecutive_losses}회")
            
        return risk_alerts

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'average_profit': 0,
            'total_trades': 0,
            'api_response_times': [],
            'execution_times': defaultdict(list)
        }
        self.trade_history = []
        
    def update_metrics(self, new_trade):
        """새로운 거래 결과로 지�� ���������이��"""
        try:
            self.trade_history.append(new_trade)
            self._calculate_metrics()
            self._log_metrics()
        except Exception as e:
            logger.error(f"지표 업데이트 중 오류: {e}")
            
    def _calculate_metrics(self):
        """모든 성과 지표 계산"""
        try:
            if not self.trade_history:
                return
                
            # 기본 지표 계산
            profits = [trade['profit'] for trade in self.trade_history]
            wins = sum(1 for p in profits if p > 0)
            losses = sum(1 for p in profits if p < 0)
            
            # 승률 계산
            self.metrics['win_rate'] = wins / len(self.trade_history) if self.trade_history else 0
            
            # 수익 팩터 계산
            total_profit = sum(p for p in profits if p > 0)
            total_loss = abs(sum(p for p in profits if p < 0))
            self.metrics['profit_factor'] = total_profit / total_loss if total_loss != 0 else float('inf')
            
            # 평균 수익 계산
            self.metrics['average_profit'] = sum(profits) / len(profits) if profits else 0
            
            # 최대 낙폭 계산
            cumulative = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            self.metrics['max_drawdown'] = np.max(drawdowns) if len(drawdowns) > 0 else 0
            
            # 샤프 비율 계산
            returns = np.array(profits)
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns, ddof=1)
                self.metrics['sharpe_ratio'] = avg_return / std_return if std_return != 0 else 0
                
            self.metrics['total_trades'] = len(self.trade_history)
            
        except Exception as e:
            logger.error(f"지표 계산 중 오류: {e}")
            
    def record_api_response_time(self, response_time):
        """API 응답 시간 기록"""
        try:
            self.metrics['api_response_times'].append(response_time)
            if len(self.metrics['api_response_times']) > 1000:  # 최근 1000개만 유지
                self.metrics['api_response_times'].pop(0)
        except Exception as e:
            logger.error(f"API 응답 시간 기록 중 오류: {e}")
            
    def record_execution_time(self, operation, execution_time):
        """실행 시간 기록"""
        try:
            self.metrics['execution_times'][operation].append(execution_time)
            if len(self.metrics['execution_times'][operation]) > 100:  # 최근 100개만 유지
                self.metrics['execution_times'][operation].pop(0)
        except Exception as e:
            logger.error(f"실행 시간 기록 중 오류: {e}")
            
    def get_performance_report(self):
        """성능 보고서 생성"""
        try:
            report = {
                'trading_metrics': {
                    'win_rate': f"{self.metrics['win_rate']:.2%}",
                    'profit_factor': f"{self.metrics['profit_factor']:.2f}",
                    'average_profit': f"{self.metrics['average_profit']:.2%}",
                    'max_drawdown': f"{self.metrics['max_drawdown']:.2%}",
                    'sharpe_ratio': f"{self.metrics['sharpe_ratio']:.2f}",
                    'total_trades': self.metrics['total_trades']
                },
                'system_metrics': {
                    'avg_api_response': f"{np.mean(self.metrics['api_response_times']):.3f}s",
                    'execution_times': {
                        op: f"{np.mean(times):.3f}s"
                        for op, times in self.metrics['execution_times'].items()
                    }
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"성능 보고서 생성 중 오류: {e}")
            return None
            
    def _log_metrics(self):
        """현재 지표들을 로깅"""
        try:
            report = self.get_performance_report()
            if report:
                logger.info("현재 성능 지표:")
                logger.info(json.dumps(report, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"지표 로깅 중 오류: {e}")

class MarketAnalyzer:
    def __init__(self):
        self.market_states = {
            'trend': None,     # 상승/하락/횡
            'volatility': None,  # /중/저
            'volume': None      # 급증/정상/저조
        }
        
    def analyze_market_state(self, df):
        # 트렌드 분석
        self.market_states['trend'] = self._analyze_trend(df)
        
        # 성 분석
        self.market_states['volatility'] = self._analyze_volatility(df)
        
        # 거래량 분���
        self.market_states['volume'] = self._analyze_volume(df)

class TradingSystemTest:
    def __init__(self):
        self.test_scenarios = [
            'normal_market',
            'high_volatility',
            'low_liquidity',
            'market_crash',
            'rapid_recovery'
        ]
        
    def run_system_tests(self):
        results = {}
        for scenario in self.test_scenarios:
            results[scenario] = self._test_scenario(scenario)
        return results

class EnhancedLogger:
    def __init__(self):
        self.log_levels = {
            'trade': logging.INFO,
            'risk': logging.WARNING,
            'error': logging.ERROR,
            'debug': logging.DEBUG
        }
        
    def setup_logging(self):
        # 파일 핸들러 설정
        file_handler = RotatingFileHandler(
            'trading.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )

class ErrorHandlingSystem:
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_thresholds = {
            'api_error': 5,
            'prediction_error': 3,
            'execution_error': 2
        }
        self.recovery_strategies = {
            'api_error': self._handle_api_error,
            'prediction_error': self._handle_prediction_error,
            'execution_error': self._handle_execution_error
        }
        
    def handle_error(self, error_type, error):
        self.error_counts[error_type] += 1
        
        if self.error_counts[error_type] >= self.error_thresholds[error_type]:
            self._execute_recovery_strategy(error_type)
            
        logger.error(f"{error_type}: {str(error)}")

class UnifiedErrorHandler:
    def __init__(self):
        self.error_handler = ErrorHandlingSystem()
        self.logger = EnhancedLogger()
        
    def handle_trading_error(self, error_type, error, context=None):
        # 러 처리
        self.error_handler.handle_error(error_type, error)
        
        # 깅
        self.logger.log_error(error_type, error, context)
        
        # 복구 전략 실행
        recovery_action = self.error_handler.get_recovery_strategy(error_type)
        if recovery_action:
            recovery_action(context)

def backtest(initial_balance: float = 500000.0):
    # 초기 설정
    balance = initial_balance  # 이 부분 추가 필
    crypto_held = 0
    portfolio_values = []
    trades = []

def analyze_time_frame_signals(df_30m, df_daily, df_weekly):
    """여러 시간대의 기술적 지표 분석"""
    try:
        signals = {}
        if df_30m is not None:
            signals['30m'] = analyze_timeframe(df_30m)
        if df_daily is not None:
            signals['daily'] = analyze_timeframe(df_daily)
        if df_weekly is not None:
            signals['weekly'] = analyze_timeframe(df_weekly)
        return signals
    except Exception as e:
        logger.error(f"시간대별 분석 중 오류: {e}")
        return {}

def ai_trading():
    try:
        # Upbit 객체 생성
        upbit = create_upbit_instance()  # 이 줄 추가
        
        if not upbit:
            raise ValueError("업비트 API 키가 정의되지 않았습니다.")

        # 데이터 수집
        logger.info("데이터 수집 중...")
        df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=100)
        
        if df_daily is None:
            raise ValueError("일봉 데이터 수집 실패")
            
        # 지표 추가
        df_daily = add_indicators(df_daily)
        if df_daily is None:
            raise ValueError("기술 지표 계산 실패")
            
        # ML 모 초기화 및 학습
        predictor = EnhancedBitcoinPredictor()
        predictor.train(df_daily)
        
        # 예측 및 신뢰도 계산
        predicted_price, ml_confidence = predictor.predict_next(df_daily)
        if predicted_price is not None:
            logger.info(f"예측 가격: {predicted_price:,.0f}원")
            logger.info(f"ML 모델 신뢰도: {ml_confidence:.2%}")
        else:
            logger.error("가격 예측 실패")
        
        # 거래 조건 확인
        trading_conditions = TradingConditions()
        should_execute, interval = trading_conditions.should_execute_trading(
            df_daily,
            upbit,  # upbit 인스턴스 추가
            ml_confidence=ml_confidence
        )
        
        if should_execute:
            logger.info("거래 조건 충족. 거래 진행...")
            # 현재 가격과 잔고 정보 가져오기
            current_price = pyupbit.get_current_price("KRW-BTC")
            balances = upbit.get_balances()
            btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
            krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
            
            # ML 예측 결과에 따른 거래 방향 결정
            decision = "buy" if predicted_price > current_price else "sell"
            
            # 거래 실행
            trade_result = execute_trade(
                TradingDecision(
                    decision=decision,
                    percentage=int(min(max(ml_confidence * 100, 10), 100)),  # 신뢰도에 따른 거래 비중
                    reason=f"ML 예측 기반 거래 (예측가격: {predicted_price:,.0f}원)",
                    confidence=ml_confidence
                ),
                upbit,
                current_price,
                btc_balance,
                krw_balance
            )
        else:
            logger.info("현재 거래 조건이 충족되지 않음")
        
    except Exception as e:
        logger.error(f"AI 트레이딩 중 오류 발생: {e}")

def continuous_monitoring(stop_event):
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    upbit = pyupbit.Upbit(access, secret)
    
    while not stop_event.is_set():
        try:
            monitor = TradingMonitor()
            current_price = pyupbit.get_current_price("KRW-BTC")
            volume = get_current_volume()
            profit = calculate_current_profit(upbit)
            
            status = monitor.check_market_conditions(current_price, volume, profit)
            if status:
                send_trading_notification("warning", 0, f"위험 감지: {status}")
                
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"모니터링 중 오류: {e}")
            time.sleep(60)

def get_scheduled_trading_times():
    return [
        "10:00",  # 아시아 시장 분석
        "18:00",  # 유럽 시장 분석
        "23:30"   # 미국 시장 분석
    ]

def monitor_technical_signals(stop_event):
    while not stop_event.is_set():
        try:
            df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=100)
            if df is None:
                continue
                
            df = add_indicators(df)
            signals = check_technical_signals(df)
            
            if signals['should_trade']:
                execute_technical_trade(signals)
                
            time.sleep(300)
            
        except Exception as e:
            logger.error(f"기술적 지표 모니터링 중 오류: {e}")
            time.sleep(60)

def check_technical_signals(df):
    signals = {
        'should_trade': False,
        'type': None,
        'reason': []
    }
    
    # RSI 과매수/과매도 체크
    current_rsi = df['rsi'].iloc[-1]
    if current_rsi <= 25:
        signals['should_trade'] = True
        signals['type'] = 'buy'
        signals['reason'].append(f"RSI 과매도: {current_rsi:.2f}")
    elif current_rsi >= 75:
        signals['should_trade'] = True
        signals['type'] = 'sell'
        signals['reason'].append(f"RSI 과수: {current_rsi:.2f}")
    
    # 볼린저 밴드 돌�� 체크
    current_price = df['close'].iloc[-1]
    upper_band = df['bb_bbh'].iloc[-1]
    lower_band = df['bb_bbl'].iloc[-1]
    
    if current_price > upper_band:
        signals['should_trade'] = True
        signals['type'] = 'sell'
        signals['reason'].append("볼린저 밴드 상단 돌파")
    elif current_price < lower_band:
        signals['should_trade'] = True
        signals['type'] = 'buy'
        signals['reason'].append("볼린저 밴드 하단 돌파")
    
    return signals

def execute_technical_trade(signals):
    """기술적 지표 기반 거래 실행"""
    try:
        # Upbit 객체 생성
        upbit = create_upbit_instance()
        if not upbit:
            return
            
        # 현재 가격 조회
        current_price = pyupbit.get_current_price("KRW-BTC")
        if current_price is None:
            return
            
        # 잔고 확인
        balances = upbit.get_balances()
        btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
        krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
        
        # 거래 금액 계산
        trade_amount = calculate_trade_amount(signals, krw_balance, btc_balance)
        
        # 거래 실행
        result = TradingDecision(
            decision=signals['type'],
            percentage=trade_amount,
            reason=", ".join(signals['reason']),
            confidence=signals['confidence']
        )
        
        execute_trade(
            result,
            upbit,
            current_price,
            btc_balance,
            krw_balance
        )
            
    except Exception as e:
        logger.error(f"기술적 지표 기반 거래 실행 중 오류: {e}")

def full_market_analysis():
    try:
        logger.info("전체 시장 분석 시작...")
        
        # 여러 ��간대 데이터 수집
        df_5m = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=100)
        df_1h = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
        df_1d = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
        
        if any(df is None for df in [df_5m, df_1h, df_1d]):
            raise ValueError("데이터 수집 실패")
        
        # 종합 분석 실행
        analysis_result = {
            'short_term': analyze_market_data(df_5m, "단��"),
            'medium_term': analyze_market_data(df_1h, "중기"),
            'long_term': analyze_market_data(df_1d, "장기")
        }
        
        # 거래 신호 확인
        if should_execute_trade(analysis_result):
            execute_analysis_based_trade(analysis_result)
            
    except Exception as e:
        logger.error(f"전체 시장 분석 중 오류: {e}")

def create_upbit_instance():
    """Upbit 인스턴스 생성"""
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    return pyupbit.Upbit(access, secret)

def calculate_trade_amount(signals, krw_balance, btc_balance):
    """거래 금액 계산"""
    try:
        # 기본 비율 설정
        base_percentage = 1.0
        
        # 신호 강도에 따른 조정
        if signals.get('confidence', 0) > 0.8:
            base_percentage *= 1.5
        
        # 시장 변동성에 따른 조정
        volatility = get_current_volatility(None)  # upbit 파라터 제거
        if volatility > 0.03:  # 높은 변동성
            base_percentage *= 0.7
            
        # 계좌 잔고에 따른 조정
        total_balance = krw_balance + (btc_balance * pyupbit.get_current_price("KRW-BTC"))
        if total_balance < 1000000:  # 100만원 미만
            base_percentage *= 0.5
            
        # 최대/최소 제한
        return min(max(base_percentage, 0.5), 5.0)
        
    except Exception as e:
        logger.error(f"거래 금액 계산 중 오류: {e}")
        return 1.0  # 오류 시 기본값

def analyze_market_data(df, timeframe):
    """시장 데이터 분석"""
    analysis = {
        'trend': None,
        'strength': 0,
        'signals': []
    }
    
    # RSI 분석
    if 'rsi' in df.columns:
        current_rsi = df['rsi'].iloc[-1]
        if current_rsi < 30:
            analysis['signals'].append('과매도')
            analysis['strength'] += 1
        elif current_rsi > 70:
            analysis['signals'].append('과매수')
            analysis['strength'] += 1
            
    # MACD 분석
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
            analysis['trend'] = 'bullish'
        else:
            analysis['trend'] = 'bearish'
            
    return analysis

def should_execute_trade(analysis_result):
    """거래 실행 여부 결정"""
    # 모든 시간대의 트렌드가 같은 방향일 때
    trends = [result['trend'] for result in analysis_result.values()]
    if len(set(trends)) == 1 and None not in trends:
        return True
        
    # 강한 신호가 있을 때
    total_strength = sum(result['strength'] for result in analysis_result.values())
    return total_strength >= 2

def execute_analysis_based_trade(analysis_result):
    """분석 기반 거래 실행"""
    try:
        upbit = create_upbit_instance()
        current_price = pyupbit.get_current_price("KRW-BTC")
        
        if current_price is None:
            return
            
        # 거래 방향 결정
        trends = [result['trend'] for result in analysis_result.values()]
        decision = 'buy' if trends.count('bullish') > trends.count('bearish') else 'sell'
        
        balances = upbit.get_balances()
        btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
        krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
        
        # 거래 실행
        trade_amount = calculate_trade_amount({'reason': []}, krw_balance, btc_balance)
        execute_trade(
            TradingDecision(
                decision=decision,
                percentage=trade_amount,
                reason="종합 시장 분석 기반 거래",
                confidence=0.7
            ),
            upbit,
            current_price,
            btc_balance,
            krw_balance
        )
        
    except Exception as e:
        logger.error(f"분석 기반 거래 실행 중 오류: {e}")

class DataCache:
    def __init__(self):
        self.cache = {}
        self.cache_time = {}
        self.expiry = {
            'minute5': 300,    # 5분
            'minute30': 1800,  # 30분
            'day': 86400      # 24시간
        }
        
    def get_data(self, interval, count):
        cache_key = f"{interval}_{count}"
        current_time = time.time()
        
        # 캐시된 데이터가 있고 유효한 경우
        if (cache_key in self.cache and 
            current_time - self.cache_time[cache_key] < self.expiry[interval]):
            return self.cache[cache_key]
            
        # 새로운 데이터 수집
        data = pyupbit.get_ohlcv("KRW-BTC", interval=interval, count=count)
        if data is not None:
            self.cache[cache_key] = data
            self.cache_time[cache_key] = current_time
            
        return data

def analyze_technical_indicators(df):
    """기술적 지표 분석"""
    try:
        signals = {
            'type': None,  # 'buy' 또는 'sell'
            'reason': [],
            'confidence': 0
        }
        
        # RSI 분석
        rsi = calculate_rsi(df['close'])
        if rsi is not None:
            current_rsi = rsi.iloc[-1]
            if current_rsi < 30:
                signals['type'] = 'buy'
                signals['reason'].append(f"RSI 과매도({current_rsi:.2f})")
                signals['confidence'] += 0.3
            elif current_rsi > 70:
                signals['type'] = 'sell'
                signals['reason'].append(f"RSI 과매수({current_rsi:.2f})")
                signals['confidence'] += 0.3
        
        # MACD 분석
        macd, signal_line = calculate_macd(df['close'])
        if macd is not None and signal_line is not None:
            if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
                signals['type'] = 'buy'
                signals['reason'].append("MACD 골든크로스")
                signals['confidence'] += 0.3
            elif macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]:
                signals['type'] = 'sell'
                signals['reason'].append("MACD 데드크로스")
                signals['confidence'] += 0.3
        
        # 볼린저 밴드 분석
        ma20, upper_band, lower_band = calculate_bollinger(df['close'])
        if all(x is not None for x in [ma20, upper_band, lower_band]):
            current_price = df['close'].iloc[-1]
            if current_price < lower_band.iloc[-1]:
                signals['type'] = 'buy'
                signals['reason'].append("볼린저 밴드 하단 돌파")
                signals['confidence'] += 0.2
            elif current_price > upper_band.iloc[-1]:
                signals['type'] = 'sell'
                signals['reason'].append("볼린저 밴드 상단 돌파")
                signals['confidence'] += 0.2
        
        return signals
        
    except Exception as e:
        logger.error(f"기술적 지표 분석 중 오류: {e}")
        return {'type': None, 'reason': [], 'confidence': 0}

def calculate_stoch_rsi(prices, period=14, smooth_k=3, smooth_d=3):
    """스토캐스틱 RSI 계산"""
    try:
        rsi = calculate_rsi(prices)
        if rsi is None:
            return None
            
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        k = stoch_rsi.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        return k
        
    except Exception as e:
        logger.error(f"스토캐스틱 RSI 계산 중 오류: {e}")
        return None

def calculate_vwap(df):
    """거래량 가중 평균가격(VWAP) 계산"""
    try:
        v = df['volume']
        tp = (df['high'] + df['low'] + df['close']) / 3
        return (tp * v).cumsum() / v.cumsum()
    except Exception as e:
        logger.error(f"VWAP 계산 중 오류: {e}")
        return None

def calculate_ml_prediction_weight(predicted_price, current_price):
    """ML 예측 가중치 계산"""
    try:
        if predicted_price is None or current_price is None:
            return 0.0
            
        # 예측 가��과 현재 가격의 차이 계산
        price_diff_percent = (predicted_price - current_price) / current_price
        
        # 가중치 계산 (시그모이드 함수 사용)
        weight = 1 / (1 + np.exp(-10 * abs(price_diff_percent)))
        
        # 측 신뢰도에 따른 가중치 조정
        if abs(price_diff_percent) < 0.01:  # 1% 미만의 변화
            weight *= 0.5
        elif abs(price_diff_percent) > 0.05:  # 5% 초과의 변���
            weight *= 0.8
            
        return weight
        
    except Exception as e:
        logger.error(f"ML 예측 가중치 계산 중 오류: {e}")
        return 0.0

def combine_signals(technical_signals, ml_weight, time_frame_signals):
    """신호 종합"""
    try:
        if not technical_signals:
            return {'action': 'hold', 'percentage': 0, 'confidence': 0}
            
        # 기본 점수 초기화
        scores = {'buy': 0, 'sell': 0}
        total_signals = 0
        
        # 기술적 지표 점수 계산
        if 'rsi' in technical_signals:
            total_signals += 1
            if technical_signals['rsi']['signal'] == '과매도':
                scores['buy'] += 1
            elif technical_signals['rsi']['signal'] == '과매수':
                scores['sell'] += 1
                
        if 'macd' in technical_signals:
            total_signals += 1
            if technical_signals['macd']['signal'] == '매수':
                scores['buy'] += 1
            else:
                scores['sell'] += 1
                
        if 'bollinger' in technical_signals:
            total_signals += 1
            if technical_signals['bollinger']['position'] == '하단돌파':
                scores['buy'] += 1
            elif technical_signals['bollinger']['position'] == '상단돌파':
                scores['sell'] += 1
                
        # ML 가중치 적용
        if ml_weight > 0.6:
            scores['buy'] += ml_weight
            total_signals += 1
        elif ml_weight < 0.4:
            scores['sell'] += (1 - ml_weight)
            total_signals += 1
            
        # 시간대별 신호 반영
        for timeframe, signals in time_frame_signals.items():
            if signals.get('trend') == 'bullish':
                scores['buy'] += 0.5
            elif signals.get('trend') == 'bearish':
                scores['sell'] += 0.5
            total_signals += 0.5
            
        # 최종 결정
        if total_signals == 0:
            return {'action': 'hold', 'percentage': 0, 'confidence': 0}
            
        buy_score = scores['buy'] / total_signals
        sell_score = scores['sell'] / total_signals
        
        # 거래 결정 및 신뢰도 계산
        if abs(buy_score - sell_score) < 0.2:  # 신호가 약할 경우
            action = 'hold'
            confidence = 0
            percentage = 0
        elif buy_score > sell_score:
            action = 'buy'
            confidence = buy_score
            percentage = min(max(confidence * 10, 1), 5)  # 1~5% 범위
        else:
            action = 'sell'
            confidence = sell_score
            percentage = min(max(confidence * 10, 1), 5)  # 1~5% 범위
            
        return {
            'action': action,
            'percentage': percentage,
            'confidence': confidence
        }
        
    except Exception as e:
        logger.error(f"신호 종합 중 오류: {e}")
        return {'action': 'hold', 'percentage': 0, 'confidence': 0}

class StrategyOptimizer:
    def __init__(self):
        self.strategies = {}
        self.performance_history = defaultdict(list)
        
    def evaluate_strategy(self, strategy_name, trade_result):
        """전략 성과 평가"""
        self.performance_history[strategy_name].append({
            'timestamp': time.time(),
            'profit': trade_result.profit,
            'success': trade_result.success,
            'market_condition': trade_result.market_condition
        })
        
    def get_best_strategy(self, market_condition):
        """현재 시장 상황에 가장 적합한 전략 선택"""
        strategy_scores = {}
        
        for name, history in self.performance_history.items():
            # 최근 100개의 거래 결과만 사용
            recent_trades = history[-100:]
            
            # 수익성 계산
            profits = [trade['profit'] for trade in recent_trades]
            avg_profit = sum(profits) / len(profits) if profits else 0
            
            # 성공률 계산
            success_rate = sum(1 for trade in recent_trades if trade['success']) / len(recent_trades)
            
            # 시장 상황 적합도 계산
            market_fit = sum(1 for trade in recent_trades 
                           if trade['market_condition'] == market_condition) / len(recent_trades)
            
            # 종합 점수 계산
            strategy_scores[name] = (avg_profit * 0.4 + success_rate * 0.4 + market_fit * 0.2)
            
        return max(strategy_scores.items(), key=lambda x: x[1])[0]

def main():
    try:
        logger.info("=== 시스템 시작 상태 ===")
        logger.info(f"현재 시간: {datetime.now()}")
        logger.info(f"스케줄된 거래 시간: {get_scheduled_trading_times()}")
        logger.info(f"모니터링 간격: 1분")
        logger.info(f"기술적 지표 모니터링 간격: 5분")
        logger.info("=====================")
        
        logger.info("비트코인 자동매매 프로그램 시작...")
        
        # 프로그램 시작시 즉시 분석 및 매매 결정 실행
        try:
            logger.info("초기 시장 분석 시작...")
            # Upbit 인스턴스 생성
            upbit = create_upbit_instance()
            if not upbit:
                raise ValueError("Upbit 인스턴스 생성 실패")
                
            df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=100)
            
            if df_daily is None:
                raise ValueError("일봉 데이터 수집 실패")
                
            # 지표 추가
            df_daily = add_indicators(df_daily)
            if df_daily is None:
                raise ValueError("기술 지표 계산 실패")
                
            # ML 모델 초기화 및 학습
            predictor = EnhancedBitcoinPredictor()
            predictor.train(df_daily)
            
            # 예측 및 신뢰도 계산
            predicted_price, ml_confidence = predictor.predict_next(df_daily)
            if predicted_price is not None:
                logger.info(f"예측 가격: {predicted_price:,.0f}원")
                logger.info(f"ML 모델 신뢰도: {ml_confidence:.2%}")
                
                # 거래 조건 확인
                trading_conditions = TradingConditions()
                should_execute, interval = trading_conditions.should_execute_trading(
                    df_daily,
                    upbit,
                    ml_confidence=ml_confidence
                )
                
                if should_execute:
                    logger.info("거래 조건 충족. 거래 진행...")
                    # 현재 가격과 잔고 정보 가져오기
                    current_price = pyupbit.get_current_price("KRW-BTC")
                    balances = upbit.get_balances()
                    btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
                    krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
                    
                    # ML 예측 결과에 따른 거래 방향 결정
                    decision = "buy" if predicted_price > current_price else "sell"
                    
                    # 거래 실행
                    trade_result = execute_trade(
                        TradingDecision(
                            decision=decision,
                            percentage=int(min(max(ml_confidence * 100, 10), 100)),  # 신뢰도에 따른 거래 비중
                            reason=f"ML 예측 기반 거래 (예측가격: {predicted_price:,.0f}원)",
                            confidence=ml_confidence
                        ),
                        upbit,
                        current_price,
                        btc_balance,
                        krw_balance
                    )
                else:
                    logger.info("현재 거래 조건이 충족되지 않음")
            
        except Exception as e:
            logger.error(f"초기 분석 중 오류 발생: {e}")
        
        # 스레드 종료 이벤트 추가
        stop_event = threading.Event()
        
        # 모니터링 스레드 시작
        monitoring_thread = threading.Thread(
            target=continuous_monitoring,
            args=(stop_event,)
        )
        monitoring_thread.daemon = True
        monitoring_thread.start()
        logger.info("모니터링 시작됨")
        
        # 기술적 지표 모니터링 스레드 시작
        technical_thread = threading.Thread(
            target=monitor_technical_signals,
            args=(stop_event,)
        )
        technical_thread.daemon = True
        technical_thread.start()
        logger.info("기술적 지표 모니터링 시작됨")
        
        # 정해진 시간에 전체 시장 분석
        for trading_time in get_scheduled_trading_times():
            schedule.every().day.at(trading_time).do(full_market_analysis)
        logger.info(f"거래 스케줄 설정됨: {get_scheduled_trading_times()}")
        
        # 1시간 주기 기본 매매 로직
        schedule.every(60).minutes.do(ai_trading)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("프로그램 종료 중...")
            stop_event.set()
            monitoring_thread.join(timeout=5)
            technical_thread.join(timeout=5)
            
    except Exception as e:
            logger.error(f"메인 함수 실행 중 오류: {e}")
            time.sleep(60)
            main()  # 오류 발생시 재시작

if __name__ == "__main__":
    main()
