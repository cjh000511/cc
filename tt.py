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
from openai import OpenAI
import sqlite3
import schedule
import sys
from logging.handlers import RotatingFileHandler

class TradingDecision(BaseModel):
    decision: str
    percentage: int
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            '/home/ec2-user/cc/trading.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
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
    
    # MACD (Moving Average Convergence Divergence) 추
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
    logger.info("ChromeDriver 설정 중...")
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
        time.sleep(2)  # 클릭 후 잠시 대기
    except TimeoutException:
        logger.error(f"{element_name} 요소를 찾는 데 시간이 초과되었습니다.")
    except ElementClickInterceptedException:
        logger.error(f"{element_name} 요소를 클릭할 수 없습니다. 다른 요소에 가려져 있을 수 있습니다.")
    except NoSuchElementException:
        logger.error(f"{element_name} 요소를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"{element_name} 클릭 중 오류 발생: {e}")

def perform_chart_actions(driver):
    # 시간 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]",
        "시간 메뉴"
    )
    
    # 1시간 옵션 선택
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
    
    # 볼린저 밴드 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[15]",
        "볼린저 밴드 옵션"
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
        
        # base64로 인코딩
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return base64_image
    except Exception as e:
        logger.error(f"스크린샷 캡처 및 인코 중 오류 발생: {e}")
        return None

def get_onchain_data():
    """
    Dune Analytics API를 통해 온체인 데이터를 가져오는 함수
    """
    try:
        headers = {
            "x-dune-api-key": os.getenv("DUNE_API_KEY")
        }
        
        # 거래소 순유입량 쿼리 (예시 리 ID)
        exchange_flow = requests.get(
            "https://api.dune.com/api/v1/query/1234567/results",
            headers=headers
        ).json()
        
        # 채굴자 지갑 데이터 쿼리
        miner_data = requests.get(
            "https://api.dune.com/api/v1/query/7654321/results",
            headers=headers
        ).json()
        
        # 해시레이트 데이터 쿼리
        hashrate_data = requests.get(
            "https://api.dune.com/api/v1/query/9876543/results",
            headers=headers
        ).json()
        
        return {
            "exchange_inflow": exchange_flow.get('result', {}).get('rows', []),
            "exchange_outflow": exchange_flow.get('result', {}).get('rows', []),
            "miner_to_exchange": miner_data.get('result', {}).get('rows', []),
            "miner_balance": miner_data.get('result', {}).get('rows', []),
            "hashrate": hashrate_data.get('result', {}).get('rows', [])
        }
    except Exception as e:
        logger.error(f"온체인 데이터 조회 중 오류 발생: {e}")
        return None

def analyze_onchain_signals(onchain_data):
    """
    온체인 데이터를 분석하여 매수/도 신호를 생성하는 함수
    """
    if not onchain_data:
        return {
            "exchange_flow_signal": "neutral",
            "miner_activity_signal": "neutral",
            "hashrate_signal": "neutral"
        }
    
    signals = {
        "exchange_flow_signal": "neutral",
        "miner_activity_signal": "neutral",
        "hashrate_signal": "neutral"
    }
    
    try:
        # 데이터가 충분한지 확인
        if onchain_data.get('exchange_inflow') and onchain_data.get('exchange_outflow'):
            inflow = onchain_data['exchange_inflow'][-1] if onchain_data['exchange_inflow'] else 0
            outflow = onchain_data['exchange_outflow'][-1] if onchain_data['exchange_outflow'] else 0
            
            if inflow and outflow:  # 둘 다 데이터가 있는 경우만
                if inflow > outflow * 1.5:
                    signals['exchange_flow_signal'] = "sell"
                elif outflow > inflow * 1.5:
                    signals['exchange_flow_signal'] = "buy"
        
        # 채굴자 활동 분석
        if (onchain_data.get('miner_to_exchange') and 
            onchain_data.get('miner_balance') and 
            len(onchain_data['miner_balance']) >= 2):
            
            miner_to_exchange = onchain_data['miner_to_exchange'][-1] if onchain_data['miner_to_exchange'] else 0
            miner_balance = onchain_data['miner_balance'][-1] if onchain_data['miner_balance'] else 0
            prev_miner_balance = onchain_data['miner_balance'][-2] if len(onchain_data['miner_balance']) > 1 else miner_balance
            
            if miner_balance > 0:  # 0으로 나누기 방지
                if miner_to_exchange > miner_balance * 0.1:
                    signals['miner_activity_signal'] = "sell"
                elif miner_balance > prev_miner_balance:
                    signals['miner_activity_signal'] = "buy"
        
        # 해시레이트 분석
        if onchain_data.get('hashrate') and len(onchain_data['hashrate']) >= 2:
            current_hashrate = onchain_data['hashrate'][-1]
            prev_hashrate = onchain_data['hashrate'][-2]
            
            if prev_hashrate > 0:  # 0으로 나누기 방지
                hashrate_change = (current_hashrate - prev_hashrate) / prev_hashrate * 100
                if hashrate_change < -10:
                    signals['hashrate_signal'] = "sell"
        
        return signals
        
    except Exception as e:
        logger.error(f"온체인 신호 분석 중 오류 발: {e}")
        return signals  # 오류 발생 시 기본값 반환

def ai_trading():
    try:
        # Upbit 객체 생성
        access = os.getenv("UPBIT_ACCESS_KEY")
        secret = os.getenv("UPBIT_SECRET_KEY")
        upbit = pyupbit.Upbit(access, secret)

        # 온체인 데이터 수집 및 분석
        onchain_data = get_onchain_data()
        onchain_signals = analyze_onchain_signals(onchain_data)
        
        # 기본 시장 데이터 수집
        df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
        df_hourly = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
        
        # 지표 추가
        if not df_daily.empty:
            df_daily = add_indicators(df_daily)
        if not df_hourly.empty:
            df_hourly = add_indicators(df_hourly)

        # 오더북 데이터
        orderbook = pyupbit.get_orderbook("KRW-BTC")
        simplified_orderbook = {
            'asks': orderbook['asks'][:5] if orderbook and 'asks' in orderbook else [],
            'bids': orderbook['bids'][:5] if orderbook and 'bids' in orderbook else []
        }

        # Fear & Greed Index
        fear_greed_index = get_fear_and_greed_index()

        # 현재 포지션 정보
        balances = upbit.get_balances()
        filtered_balances = [balance for balance in balances if balance['currency'] in ['BTC', 'KRW']]

        # 시장 데이터 구조화
        market_data = {
            'daily_data': df_daily.to_dict() if not df_daily.empty else {},
            'hourly_data': df_hourly.to_dict() if not df_hourly.empty else {},
            'current_price': pyupbit.get_current_price("KRW-BTC"),
            'orderbook': simplified_orderbook,
            'fear_greed_index': fear_greed_index,
            'onchain_signals': onchain_signals
        }

        # AI 분석을 위한 데이터 준비
        daily_data = {
            'close': df_daily['close'].iloc[-1],
            'rsi': df_daily['rsi'].iloc[-1],
            'macd': df_daily['macd'].iloc[-1],
            'bb_upper': df_daily['bb_bbh'].iloc[-1],
            'bb_lower': df_daily['bb_bbl'].iloc[-1],
            'sma_20': df_daily['sma_20'].iloc[-1],
            'ema_12': df_daily['ema_12'].iloc[-1]
        } if not df_daily.empty else {}

        hourly_data = {
            'close': df_hourly['close'].iloc[-1],
            'rsi': df_hourly['rsi'].iloc[-1],
            'macd': df_hourly['macd'].iloc[-1],
            'bb_upper': df_hourly['bb_bbh'].iloc[-1],
            'bb_lower': df_hourly['bb_bbl'].iloc[-1]
        } if not df_hourly.empty else {}

        # 피보나치 레벨 계산
        fib_levels = fibonacci_retracement(df_daily)
        pattern = head_and_shoulders(df_daily)

        # OpenAI API 호출
        client = OpenAI()
        conn = get_db_connection()
        recent_trades = get_recent_trades(conn)
        reflection = generate_reflection(recent_trades, market_data)

        # AI 분석 요청
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert cryptocurrency trading analyst. 
                    Provide your analysis in the following JSON format:
                    {
                        "decision": "buy/sell/hold",
                        "percentage": <integer between 0-100>,
                        "reason": "<detailed analysis>"
                    }"""
                },
                {
                    "role": "user",
                    "content": f"""Technical Analysis Data:
                    1. Fibonacci Levels: {json.dumps(fib_levels) if fib_levels else 'None'}
                    2. On-Chain Signals: {json.dumps(onchain_signals)}
                    3. Current Price Position:
                       - Daily Indicators: {json.dumps(daily_data)}
                       - Hourly Indicators: {json.dumps(hourly_data)}
                    4. Pattern Detection: {pattern}
                    5. Market Context:
                       - Fear and Greed Index: {json.dumps(fear_greed_index)}
                       - Order Book Depth: {json.dumps(simplified_orderbook)}
                    6. Current Position: {json.dumps(filtered_balances)}"""
                }
            ],
            max_tokens=4095
        )

        # 응답 처리 개선
        try:
            result_text = response.choices[0].message.content.strip()
            # JSON 형식이 아닌 텍스트가 포함되어 있을 수 있으므로 정제
            if result_text.find('{') != -1:
                result_text = result_text[result_text.find('{'):result_text.rfind('}')+1]
            result_json = json.loads(result_text)
            
            result = TradingDecision(
                decision=result_json.get('decision', 'hold'),
                percentage=result_json.get('percentage', 0),
                reason=result_json.get('reason', 'No reason provided')
            )
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.error(f"AI 응답 파싱 오류: {e}")
            result = TradingDecision(
                decision='hold',
                percentage=0,
                reason='Error parsing AI response'
            )

        # 거래 행 로직...
        # (기존 거래 실행 코드를 여기에 추가)

    except Exception as e:
        logger.error(f"AI trading 실행 중 오류 발생: {e}")
        raise

    print(f"### AI Decision: {result.decision.upper()} ###")
    print(f"### Reason: {result.reason} ###")

    order_executed = False

    if result.decision == "buy":
        my_krw = upbit.get_balance("KRW")
        buy_amount = my_krw * (result.percentage / 100) * 0.9995  # 수수료 고려
        if buy_amount > 5000:
            print(f"### Buy Order Executed: {result.percentage}% of available KRW ###")
            order = upbit.buy_market_order("KRW-BTC", buy_amount)
            if order:
                order_executed = True
            print(order)
        else:
            print("### Buy Order Failed: Insufficient KRW (less than 5000 KRW) ###")
    elif result.decision == "sell":
        my_btc = upbit.get_balance("KRW-BTC")
        sell_amount = my_btc * (result.percentage / 100)
        current_price = pyupbit.get_current_price("KRW-BTC")
        if sell_amount * current_price > 5000:
            print(f"### Sell Order Executed: {result.percentage}% of held BTC ###")
            order = upbit.sell_market_order("KRW-BTC", sell_amount)
            if order:
                order_executed = True
            print(order)
        else:
            print("### Sell Order Failed: Insufficient BTC (less than 5000 KRW worth) ###")

    # 거래 실행 여부와 관계없이 현재 잔고 조회
    time.sleep(1)  # API 호출 제한을 고려하여 잠시 대기
    balances = upbit.get_balances()
    btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
    krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
    btc_avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'BTC'), 0)
    current_btc_price = pyupbit.get_current_price("KRW-BTC")

    # 거래 정보 및 반성 내용 로깅
    log_trade(conn, result.decision, result.percentage if order_executed else 0, result.reason, 
              btc_balance, krw_balance, btc_avg_buy_price, current_btc_price, reflection)

    # 데이터베이스 연결 종료
    conn.close()

def run_trading_job():
    try:
        # API 요청 제한 확인
        if check_api_limits():
            ai_trading()
        else:
            logger.warning("API 호출 한도 도달. 다음 주기까지 대기")
            time.sleep(60)
    except requests.exceptions.RequestException as e:
        logger.error(f"네트워크 오류: {e}")
        time.sleep(300)  # 5분 대기
    except Exception as e:
        logger.error(f"Trading job 실행 중 오류 발생: {e}")
        time.sleep(300)

def check_api_limits():
    """API 호출 한도 확인"""
    try:
        remaining = int(requests.get("https://api.upbit.com/v1/status/remaining").json()["remaining"])
        return remaining > 10  # 최소 10개의 요청 여유 확보
    except:
        return True  # 확인 실패시 기본적으로 진행

def main():
    try:
        # 서버 시작 시 기존 거래 상태 확인
        check_existing_positions()
        
        # 스케줄 설정
        schedule.every().day.at("03:00").do(run_trading_job)
        schedule.every().day.at("10:00").do(run_trading_job)
        schedule.every().day.at("16:00").do(run_trading_job)
        schedule.every().day.at("22:00").do(run_trading_job)
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)
            except Exception as e:
                logger.error(f"스케줄러 오류: {e}")
                time.sleep(300)
                
    except Exception as e:
        logger.error(f"Main 실행 중 치명적 오류: {e}")
        sys.exit(1)

def check_existing_positions():
    """서버 재시작 시 기존 포지션 확인"""
    try:
        upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))
        if not upbit:
            logger.error("Upbit 객체 생성 실패")
            return

        balances = upbit.get_balances()
        if not balances:
            logger.info("보유 자산이 없습니다.")
            return

        for balance in balances:
            try:
                if isinstance(balance, dict):  # dictionary 타입인지 확인
                    currency = balance.get('currency', 'UNKNOWN')
                    amount = balance.get('balance', '0')
                    logger.info(f"보유 자산: {currency} - {amount}")
                else:
                    logger.error(f"잘못된 잔고 데이터 형식: {balance}")
            except Exception as e:
                logger.error(f"잔고 데이터 처리 중 오류: {e}")

    except Exception as e:
        logger.error(f"포지션 확인 중 오류: {e}")
        logger.error(f"API 키 확인 필요: ACCESS_KEY={bool(os.getenv('UPBIT_ACCESS_KEY'))}, SECRET_KEY={bool(os.getenv('UPBIT_SECRET_KEY'))}")

if __name__ == "__main__":
    main()
