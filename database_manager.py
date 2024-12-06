import sqlite3
from contextlib import contextmanager
from typing import Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = 'bitcoin_trades.db'):
        self.db_path = db_path
        self.connection_pool = []
        self.max_connections = 5
        self.init_db()
    
    def init_db(self):
        """데이터베이스 초기화 및 테이블 생성"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    decision TEXT,
                    confidence REAL,
                    profit_loss REAL,
                    market_condition TEXT,
                    additional_info TEXT
                )
            ''')
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 관리"""
        connection = None
        try:
            connection = sqlite3.connect(self.db_path)
            yield connection
        except Exception as e:
            logger.error(f"데이터베이스 연결 중 오류: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def record_trade(self, decision: str, confidence: float, profit_loss: float, 
                    market_condition: str, additional_info: Optional[str] = None):
        """거래 기록 저장"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trading_metrics 
                    (timestamp, decision, confidence, profit_loss, market_condition, additional_info)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(),
                    decision,
                    confidence,
                    profit_loss,
                    market_condition,
                    additional_info
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"거래 기록 저장 중 오류: {e}")
            raise 
    
    def close(self):
        """데이터베이스 연결 종료"""
        try:
            for conn in self.connection_pool:
                if conn and not conn.closed:
                    conn.close()
            self.connection_pool.clear()
        except Exception as e:
            logger.error(f"DB 연결 종료 중 오류: {e}")
    
    def __del__(self):
        """소멸자에서 연결 종료"""
        self.close()