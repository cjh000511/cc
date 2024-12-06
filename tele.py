import os
from dotenv import load_dotenv
import telebot
import schedule
import time
import logging
from datetime import datetime
import sqlite3
from deep_translator import GoogleTranslator
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import pandas as pd
import re
import mplfinance as mpf
import matplotlib.pyplot as plt

# 환경 변수 로드
load_dotenv()

# 텔레그램 봇 토큰 확인
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
    raise ValueError("텔레그램 봇 토큰 또는 채팅 ID가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Telegram 봇 초기화
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

def escape_special_chars(text):
    """텔레그램 메시지의 특수문자 이스케이프 처리"""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    escaped_text = text
    for char in special_chars:
        escaped_text = escaped_text.replace(char, f'\\{char}')
    return escaped_text

def send_trading_notification(decision, percentage, detailed_reason, order_result=None):
    try:
        current_time = datetime.now().strftime('%Y\\-%m\\-%d %H:%M:%S')
        
        decision_emoji = {
            "buy": "🔵 매수",
            "sell": "🔴 매도",
            "hold": "⚪ 관망",
            "error": "⚠️ 오류",
            "거래 조건 분석": "📊 거래 조건 분석"
        }.get(decision, "")
        
        # 메시지 구성
        if decision == "거래 조건 분석":
            message = f"{detailed_reason}"
        else:
            message_parts = [
                f"{decision_emoji}",
                f"비중: {percentage}%" if percentage > 0 else None,
                f"{detailed_reason}" if detailed_reason else None
            ]
            message = "\n".join(filter(None, message_parts))

        # 메시지 전송
        bot.send_message(
            CHAT_ID,
            escape_special_chars(message),
            parse_mode='MarkdownV2'
        )
        logger.info("Telegram 알림 전송 완료")
        
    except Exception as e:
        logger.error(f"트레이딩 알림 전송 중 오류: {e}")

def send_enhanced_notification(decision, percentage, detailed_reason, performance_metrics=None, df_daily=None, order_result=None):
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 결정에 따른 이모지
        decision_emoji = {
            "buy": "🔵",
            "sell": "🔴",
            "hold": "⚪",
            "오류": "⚠️"
        }.get(decision, "⚪")
        
        # 기본 메시지 구성
        message_parts = [
            f"🕒 {current_time}",
            f"{decision_emoji} {decision}",
            f"분석: {detailed_reason}"
        ]
        
        message = "\n".join(message_parts)
        
        # 메시지 전송
        bot.send_message(
            CHAT_ID,
            message,
            parse_mode='MarkdownV2'
        )
        logger.info("Telegram 알림 전송 완료")
        
    except Exception as e:
        logger.error(f"트레이딩 알림 전송 중 오류: {e}")

def extract_brief_reason(detailed_reason):
    """상세 분석에서 핵심 내용 추출"""
    try:
        # 주요 키워드와 관련된 문장 찾기
        keywords = ['RSI', 'MACD', 'MA', '볼린저밴드', '거래량', '추세', '지지선', '저항선']
        sentences = detailed_reason.split('.')
        
        for sentence in sentences:
            for keyword in keywords:
                if keyword in sentence:
                    # 문장 정제
                    clean_sentence = sentence.strip()
                    # 문장이 너무 길면 잘라내기
                    if len(clean_sentence) > 50:
                        clean_sentence = clean_sentence[:47] + "..."
                    return clean_sentence
        
        # 키워드를 찾지 못한 경우 첫 문장 반환
        first_sentence = sentences[0].strip()
        if len(first_sentence) > 50:
            first_sentence = first_sentence[:47] + "..."
        return first_sentence
        
    except Exception as e:
        logger.error(f"간단한 이유 추출 중 오류: {e}")
        return "기술적 분석 기반"

def generate_analysis_chart(df_daily):
    chart_path = 'temp_chart.png'
    try:
        # 차트 스타일 설정
        style = mpf.make_mpf_style(
            base_mpf_style='charles',
            gridstyle='',
            y_on_right=True,
            marketcolors=mpf.make_marketcolors(
                up='red',
                down='blue',
                edge='inherit',
                wick='inherit',
                volume='in',
                ohlc='inherit'
            )
        )
        
        # 차트 생성
        fig, axes = mpf.plot(
            df_daily,
            type='candle',
            volume=True,
            style=style,
            addplot=[
                mpf.make_addplot(df_daily['sma_20'], color='red', width=0.7),
                mpf.make_addplot(df_daily['sma_50'], color='blue', width=0.7),
                mpf.make_addplot(df_daily['bb_bbm'], color='gray', width=0.7),
                mpf.make_addplot(df_daily['bb_bbh'], color='gray', linestyle='--'),
                mpf.make_addplot(df_daily['bb_bbl'], color='gray', linestyle='--')
            ],
            returnfig=True
        )
        
        # 차트 저장
        plt.savefig(chart_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return chart_path
        
    except Exception as e:
        logger.error(f"차트 생성 중 오류 발생: {e}")
        # 러 발생 시 임시 파일이 존재하면 삭제
        if os.path.exists(chart_path):
            try:
                os.remove(chart_path)
            except Exception as del_error:
                logger.error(f"임시 파일 삭제 중 오류: {del_error}")
        return None