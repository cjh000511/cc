import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class RiskParameters:
    max_position_size: float = 0.1  # 전체 자산의 최대 10%
    stop_loss_pct: float = 0.02     # 2% 손절
    take_profit_pct: float = 0.05   # 5% 익절
    max_daily_loss: float = 0.05    # 일일 최대 손실 5%
    trailing_stop_pct: float = 0.01  # 1% 트레일링 스탑

class RiskManager:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.daily_loss = 0
        self.params = RiskParameters()
        self.trailing_high = 0
        
    def calculate_position_size(self, current_balance: float, current_price: float) -> float:
        """포지션 크기 계산"""
        max_position = current_balance * self.params.max_position_size
        return min(max_position, current_balance)
    
    def should_stop_trading(self, current_balance: float) -> bool:
        """일일 손실 한도 체크"""
        daily_loss_pct = (self.initial_balance - current_balance) / self.initial_balance
        return daily_loss_pct >= self.params.max_daily_loss
    
    def update_trailing_stop(self, current_price: float, position_type: str) -> Optional[float]:
        """트레일링 스탑 업데이트"""
        if position_type == 'long':
            if current_price > self.trailing_high:
                self.trailing_high = current_price
            stop_price = self.trailing_high * (1 - self.params.trailing_stop_pct)
            return stop_price if current_price <= stop_price else None
        return None

    def get_stop_loss_price(self, entry_price: float, position_type: str) -> float:
        """손절가 계산"""
        if position_type == 'long':
            return entry_price * (1 - self.params.stop_loss_pct)
        return entry_price * (1 + self.params.stop_loss_pct)
    
    def get_take_profit_price(self, entry_price: float, position_type: str) -> float:
        """익절가 계산"""
        if position_type == 'long':
            return entry_price * (1 + self.params.take_profit_pct)
        return entry_price * (1 - self.params.take_profit_pct) 

class EnhancedRiskManager:
    def __init__(self, total_balance):
        self.total_balance = total_balance
        self.max_daily_loss = total_balance * 0.02  # 일일 최대 손실 2%
        self.position_limits = {
            'high_risk': 0.1,    # 고위험 상황 10%
            'medium_risk': 0.2,  # 중위험 상황 20%
            'low_risk': 0.3      # 저위험 상황 30%
        }
        
    def calculate_risk_level(self, volatility, market_trend, ml_confidence):
        """시장 상황별 리스크 레벨 계산"""
        risk_factors = {
            'volatility': self.normalize_volatility(volatility),
            'market_trend': self.analyze_market_trend(market_trend),
            'ml_confidence': ml_confidence
        }
        
        risk_score = sum(risk_factors.values()) / len(risk_factors)
        
        if risk_score > 0.7:
            return 'high_risk'
        elif risk_score > 0.4:
            return 'medium_risk'
        return 'low_risk'
        
    def get_position_size(self, risk_level, current_balance):
        """리스크 레벨에 따른 포지션 크기 계산"""
        max_position = self.position_limits[risk_level] * current_balance
        return min(max_position, current_balance * 0.3)  # 최대 30% 제한 