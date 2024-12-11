import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv
from telegram import Update
import numpy as np

load_dotenv()

async def analyze_elliott_wave(df):
    # 엘리엇 파동 분석
    prices = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    # 주요 변곡점 찾기 (피크와 저점)
    peaks = []
    troughs = []
    
    # 변동성 기준 설정 (노이즈 필터링)
    volatility = np.std(prices) * 0.1
    
    for i in range(2, len(prices)-2):
        # 피크 찾기 (이전 2개, 이후 2개 보다 높은 지점)
        if all(prices[i] > prices[i-j] for j in range(1,3)) and \
           all(prices[i] > prices[i+j] for j in range(1,3)) and \
           prices[i] - min(prices[i-2:i+3]) > volatility:
            peaks.append((i, prices[i]))
        
        # 저점 찾기 (이전 2개, 이후 2개 보다 낮은 지점)
        if all(prices[i] < prices[i-j] for j in range(1,3)) and \
           all(prices[i] < prices[i+j] for j in range(1,3)) and \
           max(prices[i-2:i+3]) - prices[i] > volatility:
            troughs.append((i, prices[i]))
    
    # 최근 8개의 변곡점 분석 (5파동 + 3파동 수정 분석)
    wave_points = sorted(peaks + troughs, key=lambda x: x[0])[-8:]
    
    if len(wave_points) >= 5:
        # 파동 특성 분석
        price_moves = []
        for i in range(1, len(wave_points)):
            price_moves.append(wave_points[i][1] - wave_points[i-1][1])
        
        # 상승/하락 패턴 분석
        up_moves = [move for move in price_moves if move > 0]
        down_moves = [move for move in price_moves if move < 0]
        
        message = "엘리엇 파동 분석:\n"
        
        if len(wave_points) >= 8:
            # 5-3 파동 패턴 분석
            if len(up_moves) > len(down_moves):
                # 상승 추세 분석
                if len(up_moves) >= 3 and up_moves[1] > up_moves[0]:  # 3파가 1파보다 긴 경우
                    message += "- 상승 5파동 진행 중\n"
                    message += "- 3파가 1파보다 강함 (강세 신호)\n"
                    if len(up_moves) >= 5:
                        message += "- 5파동 완성 후 조정 예상\n"
                else:
                    message += "- 상승 파동 진행 중 (파동 수: {})\n".format(len(up_moves))
            else:
                # 하락 추세 분석
                if len(down_moves) >= 3 and abs(down_moves[1]) > abs(down_moves[0]):
                    message += "- 하락 5파동 진행 중\n"
                    message += "- 3파가 1파보다 강함 (약세 신호)\n"
                    if len(down_moves) >= 5:
                        message += "- 5파동 완성 후 반등 예상\n"
                else:
                    message += "- 하락 파동 진행 중 (파동 수: {})\n".format(len(down_moves))
            
            # 파동 위치 추정
            current_price = prices[-1]
            last_two_moves = price_moves[-2:]
            if len(last_two_moves) >= 2:
                if all(move > 0 for move in last_two_moves):
                    message += "- 현재 상승 파동 진행 중\n"
                elif all(move < 0 for move in last_two_moves):
                    message += "- 현재 하락 파동 진행 중\n"
                else:
                    message += "- 파동 전환 구간\n"
        else:
            message += "- 파동 패턴 불명확 (변곡점 부족)\n"
        
        return message
    else:
        return "파동 패턴 분석을 위한 충분한 데이터가 없습니다."

async def detect_chart_patterns(df):
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    patterns = []
    
    window = 20
    recent_close = close[-window:]
    recent_high = high[-window:]
    recent_low = low[-window:]
    
    # 이중 바닥 패턴 (Double Bottom)
    if len(recent_low) >= window:
        bottoms = []
        for i in range(1, len(recent_low)-1):
            if recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i+1]:
                bottoms.append((i, recent_low[i]))
        if len(bottoms) >= 2:
            if abs(bottoms[-1][1] - bottoms[-2][1]) / bottoms[-1][1] < 0.02:  # 2% 이내 차이
                patterns.append(("이중 바닥", 2))  # (패턴, 매수 강도)
    
    # 이중 천장 패턴 (Double Top)
    if len(recent_high) >= window:
        tops = []
        for i in range(1, len(recent_high)-1):
            if recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i+1]:
                tops.append((i, recent_high[i]))
        if len(tops) >= 2:
            if abs(tops[-1][1] - tops[-2][1]) / tops[-1][1] < 0.02:
                patterns.append(("이중 천장", -2))  # (패턴, 매도 강도)
    
    # 상승 삼각형
    if len(recent_close) >= window:
        higher_lows = all(recent_low[i] > recent_low[i-1] for i in range(1, len(recent_low)))
        flat_highs = abs(max(recent_high) - min(recent_high)) / max(recent_high) < 0.02
        if higher_lows and flat_highs:
            patterns.append(("상승 삼각형", 1))
    
    # 하락 삼각형
    if len(recent_close) >= window:
        lower_highs = all(recent_high[i] < recent_high[i-1] for i in range(1, len(recent_high)))
        flat_lows = abs(max(recent_low) - min(recent_low)) / max(recent_low) < 0.02
        if lower_highs and flat_lows:
            patterns.append(("하락 삼각형", -1))
    
    # 헤드앤숄더 패턴 (Head and Shoulders)
    if len(recent_high) >= window:
        peaks = []
        for i in range(1, len(recent_high)-1):
            if recent_high[i] > recent_high[i-1] and recent_high[i] > recent_high[i+1]:
                peaks.append((i, recent_high[i]))
        if len(peaks) >= 3:
            # 중간 피크가 가장 높고, 좌우 피크가 비슷한 높이인 경우
            if peaks[1][1] > peaks[0][1] and peaks[1][1] > peaks[2][1]:
                if abs(peaks[0][1] - peaks[2][1]) / peaks[0][1] < 0.03:  # 3% 이내 차이
                    patterns.append(("헤드앤숄더", -2))  # 강한 매도 신호
    
    # 역헤드앤숄더 패턴 (Inverse Head and Shoulders)
    if len(recent_low) >= window:
        troughs = []
        for i in range(1, len(recent_low)-1):
            if recent_low[i] < recent_low[i-1] and recent_low[i] < recent_low[i+1]:
                troughs.append((i, recent_low[i]))
        if len(troughs) >= 3:
            if troughs[1][1] < troughs[0][1] and troughs[1][1] < troughs[2][1]:
                if abs(troughs[0][1] - troughs[2][1]) / troughs[0][1] < 0.03:
                    patterns.append(("역헤드앤숄더", 2))  # 강한 매수 신호
    
    # 컵앤핸들 패턴 (Cup and Handle)
    if len(recent_close) >= window:
        mid_point = len(recent_close) // 2
        left_cup = recent_close[:mid_point]
        right_cup = recent_close[mid_point:]
        
        if (max(left_cup) - min(left_cup)) / max(left_cup) < 0.1:  # 완만한 U자 형태
            if (max(right_cup) - min(right_cup)) / max(right_cup) < 0.05:  # 핸들 부분
                patterns.append(("컵앤핸들", 2))  # 강한 매수 신호
    
    # 상승 플래그 (Bull Flag)
    if len(recent_close) >= window:
        trend = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
        if trend > 0:  # 상승 추세
            volatility = np.std(recent_close) / np.mean(recent_close)
            if volatility < 0.02:  # 낮은 변동성
                patterns.append(("상승 플래그", 1))
    
    # 하락 플래그 (Bear Flag)
    if len(recent_close) >= window:
        trend = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
        if trend < 0:  # 하락 추세
            volatility = np.std(recent_close) / np.mean(recent_close)
            if volatility < 0.02:
                patterns.append(("하락 플래그", -1))
    
    # 삼각수렴 (Triangle Convergence)
    if len(recent_close) >= window:
        highs_trend = np.polyfit(range(len(recent_high)), recent_high, 1)[0]
        lows_trend = np.polyfit(range(len(recent_low)), recent_low, 1)[0]
        if abs(highs_trend) < 0.001 and abs(lows_trend) < 0.001:
            patterns.append(("삼각수렴", 1))
    
    # 쐐기형 패턴 (Wedge)
    if len(recent_close) >= window:
        highs_trend = np.polyfit(range(len(recent_high)), recent_high, 1)[0]
        lows_trend = np.polyfit(range(len(recent_low)), recent_low, 1)[0]
        if highs_trend < 0 and lows_trend < 0 and highs_trend != lows_trend:
            patterns.append(("하락 쐐기", 2))  # 반전 신호
        elif highs_trend > 0 and lows_trend > 0 and highs_trend != lows_trend:
            patterns.append(("상승 쐐기", -2))  # 반전 신호
    
    return patterns

async def get_overall_opinion(df):
    # 기존 점수 계산
    rsi = df['RSI'].iloc[-1]
    rsi_score = -2 if rsi > 80 else -1 if rsi > 70 else 2 if rsi < 20 else 1 if rsi < 30 else 0
    
    macd_value = df['MACD'].iloc[-1]
    macd_signal = df['Signal'].iloc[-1]
    macd_score = 1 if macd_value > macd_signal else -1
    
    df['MA20'] = df['Close'].rolling(window=20).mean()
    trend_score = 1 if df['Close'].iloc[-1] > df['MA20'].iloc[-1] else -1
    
    # 차트 패턴 점수 추가
    patterns = await detect_chart_patterns(df)
    pattern_score = sum(score for _, score in patterns)
    
    # 종합 점수 계산 (차트 패턴 가중치 추가)
    total_score = rsi_score + macd_score + trend_score + pattern_score
    
    # 점수에 따른 의견
    if total_score >= 4:
        return "적극매수"
    elif total_score > 0:
        return "매수"
    elif total_score == 0:
        return "홀드"
    elif total_score > -4:
        return "매도"
    else:
        return "적극매도"

async def analyze_fibonacci_levels(df):
    # 최근 고점과 저점 찾기
    high = df['High'].max()
    low = df['Low'].min()
    current_price = df['Close'].iloc[-1]
    diff = high - low
    
    # 피보나치 레벨 계산
    levels = {
        '1': high,  # 100% (고점)
        '0.786': high - diff * 0.786,  # 78.6%
        '0.618': high - diff * 0.618,  # 61.8%
        '0.5': high - diff * 0.5,    # 50%
        '0.382': high - diff * 0.382,  # 38.2%
        '0.236': high - diff * 0.236,  # 23.6%
        '0': low  # 0% (저점)
    }
    
    # 현재가 위치 분석 수정
    current_level = None
    next_level = None
    sorted_levels = sorted(levels.items(), key=lambda x: float(x[0]), reverse=True)
    
    for i in range(len(sorted_levels)-1):
        upper = sorted_levels[i]
        lower = sorted_levels[i+1]
        if upper[1] >= current_price >= lower[1]:
            current_level = lower
            next_level = upper
            break
    
    # 분석 메시지 생성
    message = "\n피보나치 되돌림 분석:\n"
    for level_name, price in sorted_levels:
        message += f"- Fib{level_name}: {price:.2f}"
        if current_level and level_name == current_level[0]:
            message += " (현재 지지선)"
        if next_level and level_name == next_level[0]:
            message += " (다음 저항선)"
        message += "\n"
    
    # 투자 시사점 추가
    if current_level and next_level:
        message += "\n투자 시사점:\n"
        message += f"- 현재 Fib{current_level[0]} ({current_level[1]:.2f}) 지지선과 "
        message += f"Fib{next_level[0]} ({next_level[1]:.2f}) 저항선 사이에서 거래 중\n"
        
        # 현재가가 어느 레벨에 더 가까운지 분석
        resistance_diff = next_level[1] - current_price
        support_diff = current_price - current_level[1]
        
        if resistance_diff < support_diff:
            message += f"- 저항선 (Fib{next_level[0]}) 돌파 시도 가능성 있음\n"
        else:
            message += f"- 지지선 (Fib{current_level[0]}) 테스트 중\n"
    
    return message, levels

async def analyze_stock(ticker):
    # 주식 데이터 가져오기
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    # RSI 계산
    df['RSI'] = df.ta.rsi()
    
    # MACD 계산
    macd = df.ta.macd()
    df['MACD'] = macd['MACD_12_26_9']
    df['Signal'] = macd['MACDs_12_26_9']
    
    # 피재가
    current_price = df['Close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    macd_value = df['MACD'].iloc[-1]
    
    # 엘리엇 파동 분석
    wave_analysis = await analyze_elliott_wave(df)
    
    # 차트 패턴 감지
    patterns = await detect_chart_patterns(df)
    pattern_text = "\n차트 패턴:\n"
    if patterns:
        for pattern, _ in patterns:
            pattern_text += f"- {pattern} 패턴 감지\n"
    else:
        pattern_text += "- 주요 패턴 없음\n"
    
    # 피보나치 분석 (기존 코드 제거하고 새 함수 사용)
    fib_message, _ = await analyze_fibonacci_levels(df)
    
    # 종합 의견
    overall_opinion = await get_overall_opinion(df)
    
    message = f"""
{ticker} 기술분석 결과:

현재가: {current_price:.2f}

RSI: {rsi:.2f}
- 과매수/과매도 상태: {'과매수' if rsi > 70 else '과매도' if rsi < 30 else '중립'}

MACD: {macd_value:.2f}
- 신호: {'매수' if macd_value > df['Signal'].iloc[-1] else '매도'}

{wave_analysis}

{pattern_text}
{fib_message}

종합 투자의견: {overall_opinion}
"""
    return message

async def handle_message(update, context):
    ticker = update.message.text.upper()
    try:
        analysis = await analyze_stock(ticker)
        await update.message.reply_text(analysis)
    except Exception as e:
        await update.message.reply_text(f"에러 발생: {str(e)}")

def main():
    # 봇 토큰 가져오기
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")

    # 애플리케이션 빌더 생성
    app = Application.builder().token(token).build()
    
    # 메시지 핸들러 등록
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # 폴링 시작
    print("봇이 시작되었습니다...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
