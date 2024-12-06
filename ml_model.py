import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import optuna
from typing import Dict, Tuple
import joblib
import os
from datetime import datetime
import logging
from transformer import CryptoTransformer

# 로거 설정
logger = logging.getLogger(__name__)

class ModelOptimizer:
    def __init__(self, predictor):
        self.predictor = predictor
        self.best_params = None
        self.best_score = float('inf')
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, X, y, n_trials=100):
        """하이퍼파라미터 최적화"""
        try:
            study = optuna.create_study(direction='minimize')
            optuna.logging.set_verbosity(optuna.logging.WARNING)  # 로그 레벨 조정
            
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
                
                self.predictor.xgb_model.set_params(**params)
                
                # 시계열 교차 검증
                tscv = TimeSeriesSplit(n_splits=5)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    self.predictor.xgb_model.fit(
                        X_train, 
                        y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                    
                    pred = self.predictor.xgb_model.predict(X_val)
                    score = mean_absolute_percentage_error(y_val, pred)
                    scores.append(score)
                
                return np.mean(scores)
            
            study.optimize(objective, n_trials=n_trials)
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            return self.best_params
            
        except Exception as e:
            self.logger.error(f"최적화 중 오류: {str(e)}")
            return None
    
    def save_model(self, model_dir='models'):
        """최적화된 모델 저장"""
        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(model_dir, f'model_{timestamp}.joblib')
            
            model_data = {
                'model': self.predictor.xgb_model,
                'params': self.best_params,
                'score': self.best_score,
                'timestamp': timestamp
            }
            
            joblib.dump(model_data, model_path)
            return True
            
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류: {str(e)}")
            return False
    
    def load_best_model(self, model_dir='models'):
        """최근 성능이 가장 좋은 모델 로드"""
        try:
            if not os.path.exists(model_dir):
                return False, "델 디렉토리가 존재하지 않습니다"
            
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
            if not model_files:
                return False, "저장된 모델이 없습니다"
            
            best_model = None
            best_score = float('inf')
            
            for model_file in model_files:
                model_path = os.path.join(model_dir, model_file)
                model_data = joblib.load(model_path)
                
                if model_data['score'] < best_score:
                    best_score = model_data['score']
                    best_model = model_data
            
            if best_model:
                self.predictor.xgb_model = best_model['model']
                self.best_params = best_model['params']
                self.best_score = best_model['score']
                return True, f"모델 로드 완료: {best_model['timestamp']}"
            
            return False, "유효한 모델을 찾을 수 없습니다"
            
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류: {str(e)}")
            return False, str(e)

class EnhancedBitcoinPredictor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sequence_length = 100
        
        # 실제 입력 차원 계산 (8개 기본 특성)
        self.feature_dim = 8
        
        # 스케일러 초기화
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        
        # 환경과 모델 초기화
        self.env = None
        self.model = None
        
        # XGBoost 모델 초기화 (그래디언트 부스팅 모델)
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,    # 트리의 개수
            learning_rate=0.1,   # 학습률
            max_depth=5,         # 트리의 최대 깊이
            random_state=42,     # 재현성을 위한 시드값
            objective='reg:squarederror'  # 회귀 문제를 위한 목적 함수
        )
        
        # 모델 최적화 도구 초기화
        self.optimizer = ModelOptimizer(self)

    def train(self, df):
        """모델 학습 with 최적화"""
        try:
            features = self._prepare_features(df)
            if features is None:
                return False
            
            # 타겟 데이터 준비 (다음 날의 종가)
            target = df['close'].shift(-1).dropna().values
            features = features[:-1]  # 마지막 행 제거하여 타겟과 크기 맞추기
            
            # 하이퍼파라미터 최적화 행
            best_params = self.optimizer.optimize(features, target)
            self.xgb_model.set_params(**best_params)
            
            # 최적화된 파라미터로 최종 학습
            self.xgb_model.fit(features, target)
            
            # 모델 저장
            self.optimizer.save_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"모델 학습 중 오류: {str(e)}")
            return False

    def predict_next(self, df):
        try:
            features = self._prepare_features(df)
            if features is None:
                return None, 0
            
            # 예측 수행
            xgb_pred = self.xgb_model.predict(features[-1:])
            
            # 현재 가격 기준 변화율 예측 (-1% ~ 1% 범위로 제한)
            predicted_change = np.clip(xgb_pred[0], -0.01, 0.01)
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_change)
            
            # 신뢰도 계산 개선
            confidence = self._calculate_confidence(features[-1:])
            confidence = min(confidence, 0.95)  # 최대 95%로 제한
            
            return predicted_price, confidence

        except Exception as e:
            self.logger.error(f"예측 중 오류: {str(e)}")
            return None, 0

    def _prepare_features(self, df):
        try:
            # 기본 특성 선택 (bb_bbh 제거)
            features = df[['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bb_bbm']].copy()
            
            # 결측값 처리
            features = features.fillna(method='ffill')
            
            # 가격 데이터와 다른 특성들 분리
            price_features = features[['open', 'high', 'low', 'close']]
            other_features = features[['volume', 'rsi', 'macd', 'bb_bbm']]
            
            # 각각 다른 스케일링 적용
            price_scaled = self.price_scaler.fit_transform(price_features)
            other_scaled = self.feature_scaler.fit_transform(other_features)
            
            # 스케일링된 데이터 결합
            scaled_features = np.hstack((price_scaled, other_scaled))
            
            return scaled_features
            
        except Exception as e:
            self.logger.error(f"특성 준비 중 오류: {str(e)}")
            return None

    def _calculate_confidence(self, features):
        try:
            confidences = []
            
            # 1. 예측 안정성 (앙상블 예측 강화)
            predictions = []
            for _ in range(20):  # 10회에서 20회로 증가
                noisy_features = features + np.random.normal(0, 0.003, features.shape)  # 노이즈 감소
                pred = self.xgb_model.predict(noisy_features)
                predictions.append(pred[0])
            
            pred_std = np.std(predictions)
            stability_confidence = 1 / (1 + 3 * pred_std)  # 가중치 조정
            confidences.append(stability_confidence)
            
            # 2. 시장 추세 강도
            price_features = features[:, :4]
            trend_strength = calculate_trend_strength(price_features)
            confidences.append(trend_strength)
            
            # 3. 모델 성능
            if hasattr(self.optimizer, 'best_score'):
                model_performance_confidence = 1 / (1 + self.optimizer.best_score)
                confidences.append(model_performance_confidence)
            
            # 4. 시장 변동성 고려
            price_volatility = np.std(price_features) / np.mean(price_features)
            volatility_confidence = 1 / (1 + 2 * price_volatility)  # 변동성 민감도 감소
            confidences.append(volatility_confidence)
            
            # 종합 신뢰도 계산 (가중치 조정)
            weights = [0.45, 0.25, 0.15, 0.15]  # 예측 안정성 가중치 증가
            final_confidence = np.average(confidences, weights=weights[:len(confidences)])
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"신뢰도 계산 중 오류: {str(e)}")
            return 0.4  # 기본 신뢰도 값 상향

def calculate_trend_strength(price_features):
    """시장 추세 강도 계산"""
    try:
        # 가격 변화율 계산
        price_changes = np.diff(price_features.flatten())
        
        # 추세의 일관성 계산
        trend_consistency = np.abs(np.sum(price_changes)) / np.sum(np.abs(price_changes))
        
        # 추세의 강도 계산
        trend_magnitude = np.mean(np.abs(price_changes))
        
        # 최종 추세 강도 (0~1 사이 값)
        trend_strength = trend_consistency * (1 - np.exp(-trend_magnitude))
        
        return min(max(trend_strength, 0), 1)
        
    except Exception as e:
        logger.error(f"추세 강도 계산 중 오류: {e}")
        return 0.5  # 오류 시 중립적 값 반환