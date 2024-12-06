import pandas as pd
import numpy as np
from ml_model import EnhancedBitcoinPredictor
import pyupbit
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def backtest(initial_balance: float = 500000.0):
    # 데이터 수집 (2023년 1월 1일부터 2024년 11월까지)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 11, 30)
    
    print(f"백테스팅 시작: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"초기 투자금: {initial_balance:,.0f}원")
    
    # 일봉 데이터 수집
    df = pyupbit.get_ohlcv("KRW-BTC", interval="day", 
                          to=end_date.strftime("%Y%m%d"), 
                          count=(end_date - start_date).days)
    
    if df is None or len(df) == 0:
        print("데이터 수집 실패")
        return
        
    # 기술적 지표 추가
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['bb_bbm'], df['bb_bbh'], df['bb_bbl'] = calculate_bollinger(df['close'])
    
    # 모델 초기화 및 백테스팅
    model = EnhancedBitcoinPredictor()
    
    # 초기 설정
    balance = initial_balance  # 현금
    crypto_held = 0           # 보유 코인
    portfolio_values = []     # 포트폴리오 가치 기록
    trades = []              # 거래 기록
    
    # 데이터 준비
    model.train(df)
    
    print("\n백테스팅 진행 중...")
    
    # sequence_length 이후부터 백테스팅 시작
    for i in range(model.sequence_length, len(df)):
        current_data = df.iloc[i-model.sequence_length:i]
        predicted_price, confidence = model.predict_next(current_data)
        
        if predicted_price is None:
            continue
            
        actual_price = df['close'].iloc[i]
        current_date = df.index[i]
        
        # 매매 전략 (신뢰도를 고려한 매매)
        if predicted_price > actual_price * 1.01 and confidence > 0.6:  # 1% 이상 상승 예측
            # 매수
            buy_amount = balance * 0.2  # 20% 매수
            if buy_amount >= 5000:  # 최소 거래금액 5000원
                crypto_held += buy_amount / actual_price
                balance -= buy_amount
                trades.append({
                    'date': current_date,
                    'type': 'buy',
                    'price': actual_price,
                    'amount': buy_amount
                })
                
        elif predicted_price < actual_price * 0.99 and confidence > 0.6:  # 1% 이상 하락 예측
            # 매도
            sell_amount = crypto_held * 0.2  # 20% 매도
            if crypto_held > 0:
                balance += sell_amount * actual_price
                crypto_held -= sell_amount
                trades.append({
                    'date': current_date,
                    'type': 'sell',
                    'price': actual_price,
                    'amount': sell_amount * actual_price
                })
        
        # 포트폴리오 가치 계산
        portfolio_value = balance + crypto_held * actual_price
        portfolio_values.append({
            'date': current_date,
            'value': portfolio_value
        })
    
    # 결과 분석 및 출력
    if len(portfolio_values) > 0:
        initial_date = portfolio_values[0]['date']
        final_date = portfolio_values[-1]['date']
        final_value = portfolio_values[-1]['value']
        total_return = (final_value - initial_balance) / initial_balance * 100
        
        print("\n=== 백테스팅 결과 ===")
        print(f"테스트 기간: {initial_date.strftime('%Y-%m-%d')} ~ {final_date.strftime('%Y-%m-%d')}")
        print(f"초기 자산: {initial_balance:,.0f}원")
        print(f"최종 자산: {final_value:,.0f}원")
        print(f"총 수익률: {total_return:.2f}%")
        print(f"총 거래 횟수: {len(trades)}회")
        
        # 월별 수익률 계산
        monthly_returns = calculate_monthly_returns(portfolio_values)
        print("\n=== 월별 수익률 ===")
        for month, return_value in monthly_returns.items():
            print(f"{month}: {return_value:.2f}%")
    
    return portfolio_values, trades

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger(prices, period=20, std=2):
    ma = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    return ma, upper, lower

def calculate_monthly_returns(portfolio_values):
    monthly_returns = {}
    prev_month = None
    prev_value = None
    
    for record in portfolio_values:
        current_month = record['date'].strftime('%Y-%m')
        if prev_month is None:
            prev_month = current_month
            prev_value = record['value']
            continue
            
        if current_month != prev_month:
            monthly_return = (record['value'] - prev_value) / prev_value * 100
            monthly_returns[prev_month] = monthly_return
            prev_month = current_month
            prev_value = record['value']
    
    return monthly_returns

if __name__ == "__main__":
    # 백테스팅 실행
    portfolio_values, trades = backtest(initial_balance=500000) 