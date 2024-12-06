import sqlite3
import logging
import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from parallel_analyzer import ParallelDataAnalyzer
from database_manager import DatabaseManager

# 로거 설정
logger = logging.getLogger(__name__)

class TradingMonitor:
    def __init__(self):
        """초기화"""
        try:
            self.db_manager = DatabaseManager()
        except Exception as e:
            logger.error(f"TradingMonitor 초기화 중 오류: {e}")
            raise

    def check_market_conditions(self, current_price, volume, profit):
        """시장 상태 체크"""
        try:
            # 급격한 가격 변동 체크
            price_change = self.check_price_volatility(current_price)
            if abs(price_change) > 0.1:  # 10% 이상 변동
                return f"급격한 가격 변동 감지: {price_change:.2%}"
            
            # 비정상적인 거래량 체크
            if volume > 0:
                avg_volume = self.get_average_volume()
                if volume > avg_volume * 3:  # 평균 거래량의 3배 초과
                    return "비정상적인 거래량 감지"
                
            # 손실 위험 체크
            if profit < -0.05:  # 5% 이상 손실
                return f"높은 손실 위험: {profit:.2%}"
            
            return None
            
        except Exception as e:
            logger.error(f"시장 상태 체크 중 오류: {e}")
            return None

    def check_price_volatility(self, current_price):
        """가격 변동성 체크"""
        try:
            df = pyupbit.get_ohlcv("KRW-BTC", interval="minute1", count=10)
            if df is None:
                return 0
            
            price_change = (current_price - df['close'].mean()) / df['close'].mean()
            return price_change
            
        except Exception as e:
            logger.error(f"가격 변동성 체크 중 오류: {e}")
            return 0

    def get_average_volume(self):
        """평균 거래량 조회"""
        try:
            df = pyupbit.get_ohlcv("KRW-BTC", interval="minute30", count=48)
            if df is not None:
                return df['volume'].mean()
            return 0
        except Exception as e:
            logger.error(f"평균 거래량 조회 중 오류: {e}")
            return 0

    def record_trade(self, decision, confidence, profit_loss):
        """거래 기록 저장"""
        try:
            market_condition = self.analyze_market_condition()
            self.db_manager.record_trade(
                decision=decision,
                confidence=confidence,
                profit_loss=profit_loss,
                market_condition=market_condition
            )
        except Exception as e:
            logger.error(f"거래 기록 저장 중 오류: {e}")

    def analyze_market_condition(self):
        """시장 상태 분석"""
        try:
            df = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=14)
            if df is None:
                return "unknown"
            
            # RSI 계산
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 볼린저 밴드 계산
            ma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            current_price = df['close'].iloc[-1]
            
            # 시장 상태 판단
            if current_rsi > 70 and current_price > upper_band.iloc[-1]:
                return "overbought"
            elif current_rsi < 30 and current_price < lower_band.iloc[-1]:
                return "oversold"
            elif current_rsi > 50 and current_price > ma20.iloc[-1]:
                return "bullish"
            elif current_rsi < 50 and current_price < ma20.iloc[-1]:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"시장 상태 분석 중 오류: {e}")
            return "unknown"

    def get_volume_analysis(self):
        """거래량 종합 분석"""
        try:
            # 여러 시간대의 거래량 데이터 수집
            volume_1h = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
            volume_1d = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
            
            if volume_1h is None or volume_1d is None:
                return None
            
            analysis = {
                'volume_trend': self._calculate_volume_trend(volume_1h),
                'volume_momentum': self._calculate_volume_momentum(volume_1h),
                'abnormal_volume': self._detect_abnormal_volume(volume_1d),
                'volume_price_correlation': self._calculate_volume_price_correlation(volume_1h)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"거래량 분석 중 오류: {e}")
            return None

    def _calculate_volume_trend(self, df):
        """거래량 추세 분석"""
        try:
            # 이동평균을 이용한 거래량 추세 계산
            volume_ma5 = df['volume'].rolling(window=5).mean()
            volume_ma20 = df['volume'].rolling(window=20).mean()
            
            # 추세 판단
            if volume_ma5.iloc[-1] > volume_ma20.iloc[-1]:
                return "상승"
            elif volume_ma5.iloc[-1] < volume_ma20.iloc[-1]:
                return "하락"
            return "중립"
            
        except Exception as e:
            logger.error(f"거래량 추세 계산 중 오류: {e}")
            return "중립"

    def _detect_abnormal_volume(self, df):
        """이상 거래량 감지"""
        try:
            # 거래량의 표준편차를 이용한 이상치 탐지
            volume_mean = df['volume'].mean()
            volume_std = df['volume'].std()
            current_volume = df['volume'].iloc[-1]
            
            z_score = (current_volume - volume_mean) / volume_std
            
            if abs(z_score) > 2:  # 2 표준편차 이상을 이상치로 간주
                return True
            return False
            
        except Exception as e:
            logger.error(f"이상 거래량 감지 중 오류: {e}")
            return False

    def __del__(self):
        """소멸자"""
        try:
            if hasattr(self, 'db_manager'):
                self.db_manager.close()
        except Exception as e:
            logger.error(f"TradingMonitor 종료 중 오류: {e}")