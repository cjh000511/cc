import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import ta
from ta.utils import dropna
import time
import requests
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel
import sqlite3
import schedule
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from collections import deque
import random
import pytz
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
import openai
import json

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Telegram 설정
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
    raise ValueError("텔레그램 봇 토큰 또는 채팅 ID가 설정되지 않았습니다. .env 파일을 확인하요.")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# 데이터베이스 모델
class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str

# ML 모델 클래스들
class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attention_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attention_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.optimizer = Adam(self.model.parameters())

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            act_values = self.model(torch.FloatTensor(state))
        return np.argmax(act_values.numpy())

class EnhancedBitcoinPredictor:
    def __init__(self, sequence_length=60, feature_dim=8):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.scaler = MinMaxScaler()
        
        # Transformer 모델 초기화
        self.transformer = TransformerBlock(
            input_dim=feature_dim,
            num_heads=4,
            hidden_dim=64
        )
        
        # RL 에이전트 초기화
        self.rl_agent = DQNAgent(
            state_size=feature_dim * sequence_length,
            action_size=3  # Buy, Sell, Hold
        )
        
        # XGBoost 모델
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )

    def train(self, df):
        """모델 학습"""
        try:
            # 데이터 전처리
            features = self._prepare_features(df)
            target = df['close'].values
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # XGBoost 모델 학습
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            return True
        except Exception as e:
            logger.error(f"모델 학습 중 오류: {e}")
            return False

    def predict_next(self, df):
        """다음 가격 예측"""
        try:
            features = self._prepare_features(df)
            xgb_pred = self.xgb_model.predict(features[-1:])
            
            # Transformer 입력 준비
            transformer_input = torch.FloatTensor(features[-self.sequence_length:])
            transformer_output = self.transformer(transformer_input.unsqueeze(0))
            
            # RL 에이전트 행동 결정
            state = features[-1].flatten()
            action = self.rl_agent.act(state)
            
            # 예측값 앙상블
            final_prediction = xgb_pred[0] * 0.6 + transformer_output.mean().item() * 0.3
            confidence = self.xgb_model.predict_proba(features[-1:])[:, action].item()
            
            return final_prediction, confidence
        except Exception as e:
            logger.error(f"예측 중 오류: {e}")
            return None, 0

    # ... (이전 메서드들 동일하게 포함)

# 트레이딩 조건 클래스
class TradingConditions:
    def __init__(self):
        self.korea_tz = pytz.timezone('Asia/Seoul')
        self.volatility_threshold = 0.03
        self.fng_thresholds = {'extreme_fear': 25, 'extreme_greed': 75}

    # ... (이전 메서드들 동일하게 포함)

    def should_execute_trading(self, df_daily, ml_confidence):
        """거래 실행 여부 확인"""
        try:
            current_time = datetime.now(self.korea_tz)
            
            # 거래시간 확인 (9:00 ~ 23:00)
            if current_time.hour < 9 or current_time.hour >= 23:
                return False, 60
            
            # 변동성 확인
            daily_volatility = self._calculate_volatility(df_daily)
            if daily_volatility < self.volatility_threshold:
                return False, 30
            
            # ML 모델 신뢰도 확인
            if ml_confidence < 0.6:
                return False, 30
            
            return True, 0
        except Exception as e:
            logger.error(f"거래 조건 확인 중 오류: {e}")
            return False, 60

    def _calculate_volatility(self, df):
        """일일 변동성 계산"""
        try:
            returns = df['close'].pct_change()
            return returns.std()
        except Exception as e:
            logger.error(f"변동성 계산 중 오류: {e}")
            return 0

# 데이터베이스 함수들
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

def log_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_price, current_price):
    """거래 내역을 데이터베이스에 기록"""
    try:
        c = conn.cursor()
        c.execute('''
            INSERT INTO trades (
                timestamp,
                decision,
                percentage,
                reason,
                btc_balance,
                krw_balance,
                btc_avg_buy_price,
                btc_krw_price,
                reflection
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            decision,
            percentage,
            reason,
            btc_balance,
            krw_balance,
            btc_avg_price,
            current_price,
            None
        ))
        conn.commit()
        logger.info("거래 내역 저장 완료")
    except Exception as e:
        logger.error(f"거래 내역 저장 중 오류: {e}")
        conn.rollback()

# 텔레그램 알림 함수
def send_trading_notification(decision, percentage, detailed_reason, order_result=None):
    try:
        markup = InlineKeyboardMarkup()
        markup.row(
            InlineKeyboardButton("거래 내역 보기", callback_data="history"),
            InlineKeyboardButton("결정 분포 보기", callback_data="distribution")
        )
        
        message = f"결정: {decision}\n비중: {percentage}%\n\n{detailed_reason}"
        bot.send_message(CHAT_ID, message, reply_markup=markup)
        logger.info("Telegram 메시지 전 완료")
    except Exception as e:
        logger.error(f"트레이딩 알림 전송 중 오류: {e}")

# 메인 트레이딩 함수
def ai_trading():
    try:
        # Upbit 객체 생성
        access = os.getenv("UPBIT_ACCESS_KEY")
        secret = os.getenv("UPBIT_SECRET_KEY")
        upbit = pyupbit.Upbit(access, secret)

        if not access or not secret:
            raise ValueError("업비트 API 키가 설정되지 않았습니다.")

        # 계좌 잔고 확인
        balances = upbit.get_balances()
        btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
        krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)

        # 데이터 수집
        logger.info("데이터 수집 중...")
        df_30m = pyupbit.get_ohlcv("KRW-BTC", interval="minute30", count=50)
        df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=100)
        df_weekly = pyupbit.get_ohlcv("KRW-BTC", interval="week", count=50)
        
        if df_30m is None or df_daily is None or df_weekly is None:
            raise ValueError("가격 데이터를 가져오는데 실패했습니다.")

        # ML 모델 초기화 및 학습
        predictor = EnhancedBitcoinPredictor()
        predictor.train(df_daily)
        
        # 예측 수행
        predicted_price, confidence = predictor.predict_next(df_daily)
        
        if predicted_price is None:
            raise ValueError("가격 예측에 실패했습니다.")

        # 현재 시장 상태 분석
        current_price = pyupbit.get_current_price("KRW-BTC")
        if current_price is None:
            raise ValueError("현재 가격을 가져오는데 실패했습니다.")

        # 시간대별 분석
        time_frame_signals = {
            '30m': analyze_timeframe(df_30m),
            'daily': analyze_timeframe(df_daily),
            'weekly': analyze_timeframe(df_weekly)
        }
        
        # 온체인 데이터 수집
        onchain_data = get_onchain_data()

        # AI 분석 수행
        analysis = generate_trading_analysis(
            current_price,
            predicted_price,
            time_frame_signals,
            onchain_data
        )
        
        # 거래 결정
        price_change = (predicted_price - current_price) / current_price * 100
        
        if price_change > 2 and krw_balance >= 5000:
            decision = "매수"
            percentage = min(int(abs(price_change) * 10), 100)
            reason = f"예상 가격 상승: {price_change:.2f}%, 신뢰도: {confidence:.2f}\n\nAI 분석:\n{analysis}"
        elif price_change < -2 and btc_balance > 0:
            decision = "매도"
            percentage = min(int(abs(price_change) * 10), 100)
            reason = f"예상 가격 하락: {price_change:.2f}%, 신뢰도: {confidence:.2f}\n\nAI 분석:\n{analysis}"
        else:
            decision = "관망"
            percentage = 0
            reason = f"변동성 부족 (변화율: {price_change:.2f}%)\n\nAI 분석:\n{analysis}"

        # 거래 실행 및 기록
        order_result = None
        if decision in ["매수", "매도"]:
            # 거래 금액 계산
            if decision == "매수":
                amount = (krw_balance * (percentage / 100)) / current_price
                if amount * current_price >= 5000:  # 최소 거래금액 확인
                    order_result = upbit.buy_market_order("KRW-BTC", amount * current_price)
            else:  # 매도
                amount = btc_balance * (percentage / 100)
                if amount * current_price >= 5000:
                    order_result = upbit.sell_market_order("KRW-BTC", amount)

        # 거래 내역 저장
        conn = init_db()
        log_trade(conn, decision, percentage, reason, btc_balance, krw_balance, 
                 upbit.get_avg_buy_price("BTC"), current_price)
        conn.close()

        # 거래 분석 및 개선점 도출
        trades_df = get_recent_trades(conn)
        current_market_data = {
            "price": current_price,
            "predicted_price": predicted_price,
            "confidence": confidence,
            "time_frame_signals": time_frame_signals,
            "onchain_data": onchain_data
        }
        reflection = generate_reflection(trades_df, current_market_data)
        
        if reflection:
            logger.info("거래 분석 결과:\n" + reflection)

        # 텔레그램 알림 전송
        send_trading_notification(decision, percentage, reason, order_result)
        
        logger.info(f"거래 완료: {decision}, {percentage}%, {reason}")

    except ValueError as e:
        logger.error(f"거래 실행 중 값 오류: {str(e)}")
        send_trading_notification("오류", 0, f"거래 중 값 오류 발생: {str(e)}")
    except Exception as e:
        logger.error(f"거래 실행 중 예상치 못한 오류: {str(e)}")
        send_trading_notification("오류", 0, f"거래 중 오류 발생: {str(e)}")

# 메인 실행 함수
def main():
    try:
        # 데이터베이스 초기화
        init_db()
        
        # 초기 실행
        ai_trading()
        
        # 스케줄러 설정
        schedule.every(1).hours.do(ai_trading)
        
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"메인 함수 실행 중 오류: {e}")
        time.sleep(60)

@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    try:
        if call.data == "history":
            conn = init_db()
            trades_df = get_recent_trades(conn)
            conn.close()
            
            if trades_df.empty:
                bot.answer_callback_query(call.id, "최근 거래 내역이 없습니다.")
                return
                
            message = "=== 최근 거래 내역 ===\n\n"
            for _, trade in trades_df.iterrows():
                message += f"시간: {trade['timestamp']}\n"
                message += f"결정: {trade['decision']}\n"
                message += f"비중: {trade['percentage']}%\n"
                message += f"이유: {trade['reason']}\n\n"
                
            bot.answer_callback_query(call.id)
            bot.send_message(call.message.chat.id, message)
            
        elif call.data == "distribution":
            conn = init_db()
            trades_df = get_recent_trades(conn)
            conn.close()
            
            if trades_df.empty:
                bot.answer_callback_query(call.id, "분석할 거래 내역이 없습니다.")
                return
                
            decision_dist = trades_df['decision'].value_counts()
            message = "=== 거래 결정 분포 ===\n\n"
            for decision, count in decision_dist.items():
                message += f"{decision}: {count}회\n"
                
            bot.answer_callback_query(call.id)
            bot.send_message(call.message.chat.id, message)
            
    except Exception as e:
        logger.error(f"콜백 쿼리 처리 중 오류: {e}")
        bot.answer_callback_query(call.id, "처리 중 오류가 발생했습니다.")

def get_recent_trades(conn, days=7):
    """최근 거래 내역 조회"""
    try:
        c = conn.cursor()
        seven_days_ago = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        
        # 최근 거래 내역 조회
        c.execute("""
            SELECT * FROM trades 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        """, (seven_days_ago,))
        
        # 결과를 DataFrame으로 변환
        columns = [description[0] for description in c.description]
        trades = c.fetchall()
        
        return pd.DataFrame(trades, columns=columns)
    
    except Exception as e:
        logger.error(f"거래 내역 조회 중 오류: {e}")
        return pd.DataFrame()  # 오류 발생시 빈 DataFrame 반환

def generate_trading_analysis(current_price, predicted_price, time_frame_signals, onchain_data):
    """OpenAI를 사용한 거래 분석 생성"""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not client.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

        price_change = ((predicted_price - current_price) / current_price) * 100
        
        prompt = f"""
        비트코인 거래 분석을 해주세요:

        현재 가격: {current_price:,}원
        예측 가격: {predicted_price:,}원 (변화율: {price_change:.2f}%)

        기술적 분석:
        - 30분봉: {time_frame_signals['30m']}
        - 일봉: {time_frame_signals['daily']}
        - 주��: {time_frame_signals['weekly']}

        온체인 데이터:
        {json.dumps(onchain_data, indent=2, ensure_ascii=False)}

        위 데이터를 바탕으로 다음을 분석해주세요:
        1. 현재 시장 상황
        2. 단기 전망
        3. 거래 추천 (매수/매도/관망)
        4. 추천 거래 비중 (0-100%)
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )

        analysis = response.choices[0].message.content
        return analysis

    except Exception as e:
        logger.error(f"거래 분석 생성 중 오류: {e}")
        return None

def get_onchain_data():
    """온체인 데이터 수집"""
    try:
        # 여기에 실제 온체인 데이터 API 호출 추가
        return {
            "miner_activity": "보통",
            "transaction_flow": "증가",
            "network_activity": "활발",
            "exchange_flows": "유출 우세"
        }
    except Exception as e:
        logger.error(f"온체인 데이터 수집 중 오류: {e}")
        return {}

def add_advanced_indicators(df):
    """고급 기술적 지표 추가"""
    try:
        df = df.copy()
        
        # 기본 기술적 지표
        df['ma7'] = ta.trend.sma_indicator(df['close'], window=7)
        df['ma25'] = ta.trend.sma_indicator(df['close'], window=25)
        df['rsi'] = ta.momentum.rsi(df['close'])
        
        # 볼린저 밴드
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.volatility.bollinger_bands(df['close'])
        
        # MACD
        df['macd'] = ta.trend.macd_diff(df['close'])
        
        # 스토캐스틱 RSI
        df['stoch_rsi'] = ta.momentum.stochrsi(df['close'])
        
        return df
    except Exception as e:
        logger.error(f"지표 추가 중 오류: {e}")
        return df

def analyze_timeframe(df):
    """시간대별 기술적 분석"""
    try:
        # RSI 분석
        rsi = df['rsi'].iloc[-1]
        rsi_signal = "과매수" if rsi > 70 else "과매도" if rsi < 30 else "중립"
        
        # MACD 분석
        macd = df['macd'].iloc[-1]
        macd_signal = "상승" if macd > 0 else "하락"
        
        # 볼린저 밴드 분석
        current_price = df['close'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        bb_signal = "상단돌파" if current_price > bb_upper else "하단돌파" if current_price < bb_lower else "밴드내"
        
        # 이동평균선 분석
        ma7 = df['ma7'].iloc[-1]
        ma25 = df['ma25'].iloc[-1]
        ma_signal = "상승추세" if ma7 > ma25 else "하락추세"
        
        return {
            "rsi": rsi_signal,
            "macd": macd_signal,
            "bollinger": bb_signal,
            "ma_trend": ma_signal
        }
    except Exception as e:
        logger.error(f"시간대 분석 중 오류: {e}")
        return {}

def calculate_performance(trades_df):
    if trades_df.empty:
        return 0
    
    initial_balance = trades_df.iloc[-1]['krw_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_krw_price']
    final_balance = trades_df.iloc[0]['krw_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_krw_price']
    
    return (final_balance - initial_balance) / initial_balance * 100

def generate_reflection(trades_df, current_market_data):
    """거래 내역 분석 및 개선점 도출"""
    try:
        performance = calculate_performance(trades_df)
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not client.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

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
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        reflection = response.choices[0].message.content
        return reflection

    except Exception as e:
        logger.error(f"거래 분석 생성 중 오류: {e}")
        return None

if __name__ == "__main__":
    main() 