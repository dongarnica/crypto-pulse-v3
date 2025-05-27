"""
Notification system with Telegram bot integration.
Sends alerts and performance updates to configured channels.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass
from enum import Enum

import httpx
from config.settings import settings

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications."""
    TRADE_EXECUTED = "trade_executed"
    PERFORMANCE_ALERT = "performance_alert"
    RISK_ALERT = "risk_alert"
    SYSTEM_ALERT = "system_alert"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_REPORT = "weekly_report"


class Priority(Enum):
    """Notification priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Notification:
    """Notification message structure."""
    message_type: NotificationType
    priority: Priority
    title: str
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    chat_id: Optional[str] = None


class TelegramNotifier:
    """
    Telegram bot for sending notifications.
    """
    
    def __init__(self):
        self.bot_token = settings.telegram_bot_token
        self.default_chat_id = settings.telegram_chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        self.enabled = bool(self.bot_token and self.default_chat_id and settings.enable_notifications)
        
        # Rate limiting
        self.last_message_time = {}
        self.message_queue = asyncio.Queue()
        self.is_processing = False
        
    async def initialize(self):
        """Initialize Telegram bot and verify connection."""
        if not self.enabled:
            logger.warning("Telegram notifications disabled - missing bot token or chat ID")
            return
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/getMe")
                if response.status_code == 200:
                    bot_info = response.json()
                    logger.info(f"Telegram bot initialized: {bot_info['result']['username']}")
                else:
                    logger.error(f"Failed to initialize Telegram bot: {response.text}")
                    self.enabled = False
                    
        except Exception as e:
            logger.error(f"Error initializing Telegram bot: {e}")
            self.enabled = False
    
    async def send_message(self, notification: Notification) -> bool:
        """Send a notification message via Telegram."""
        if not self.enabled:
            return False
        
        try:
            # Add to queue for rate-limited processing
            await self.message_queue.put(notification)
            
            # Start processing if not already running
            if not self.is_processing:
                asyncio.create_task(self._process_message_queue())
            
            return True
            
        except Exception as e:
            logger.error(f"Error queuing Telegram message: {e}")
            return False
    
    async def _process_message_queue(self):
        """Process queued messages with rate limiting."""
        if self.is_processing:
            return
        
        self.is_processing = True
        
        try:
            while not self.message_queue.empty():
                notification = await self.message_queue.get()
                
                # Rate limiting - max 1 message per 3 seconds
                now = datetime.utcnow()
                if self.last_message_time.get('default', datetime.min) + timedelta(seconds=3) > now:
                    await asyncio.sleep(3)
                
                success = await self._send_telegram_message(notification)
                if success:
                    self.last_message_time['default'] = datetime.utcnow()
                
                # Brief delay between messages
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error processing message queue: {e}")
        finally:
            self.is_processing = False
    
    async def _send_telegram_message(self, notification: Notification) -> bool:
        """Send actual Telegram message."""
        try:
            chat_id = notification.chat_id or self.default_chat_id
            
            # Format message
            formatted_message = self._format_message(notification)
            
            # Prepare request
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": formatted_message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload)
                
                if response.status_code == 200:
                    logger.debug(f"Telegram message sent: {notification.title}")
                    return True
                else:
                    logger.error(f"Failed to send Telegram message: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def _format_message(self, notification: Notification) -> str:
        """Format notification for Telegram."""
        priority_emoji = {
            Priority.LOW: "ℹ️",
            Priority.MEDIUM: "⚠️",
            Priority.HIGH: "🚨",
            Priority.CRITICAL: "🔴"
        }
        
        type_emoji = {
            NotificationType.TRADE_EXECUTED: "💰",
            NotificationType.PERFORMANCE_ALERT: "📊",
            NotificationType.RISK_ALERT: "⚠️",
            NotificationType.SYSTEM_ALERT: "🔧",
            NotificationType.DAILY_SUMMARY: "📈",
            NotificationType.WEEKLY_REPORT: "📋"
        }
        
        emoji = f"{priority_emoji.get(notification.priority, '📢')} {type_emoji.get(notification.message_type, '📢')}"
        timestamp = notification.timestamp.strftime("%H:%M:%S UTC")
        
        message = f"{emoji} <b>{notification.title}</b>\n"
        message += f"🕐 {timestamp}\n\n"
        message += notification.message
        
        # Add data if present
        if notification.data:
            message += "\n\n<b>Details:</b>"
            for key, value in notification.data.items():
                if isinstance(value, float):
                    if key in ['price', 'value', 'pnl', 'allocation']:
                        message += f"\n• {key.replace('_', ' ').title()}: ${value:.2f}"
                    elif key in ['percentage', 'return', 'ratio']:
                        message += f"\n• {key.replace('_', ' ').title()}: {value:.2%}"
                    else:
                        message += f"\n• {key.replace('_', ' ').title()}: {value:.4f}"
                else:
                    message += f"\n• {key.replace('_', ' ').title()}: {value}"
        
        return message


class NotificationManager:
    """
    Main notification manager coordinating all notification channels.
    """
    
    def __init__(self):
        self.telegram = TelegramNotifier()
        self.notification_history: List[Notification] = []
        self.enabled_types = set(NotificationType)
        self.min_priority = Priority.LOW
        
        # Message throttling
        self.throttle_rules = {
            NotificationType.TRADE_EXECUTED: timedelta(seconds=30),
            NotificationType.PERFORMANCE_ALERT: timedelta(minutes=15),
            NotificationType.RISK_ALERT: timedelta(minutes=5),
            NotificationType.SYSTEM_ALERT: timedelta(minutes=10),
        }
        self.last_sent = {}
        
    async def initialize(self):
        """Initialize notification system."""
        try:
            logger.info("Initializing notification system")
            await self.telegram.initialize()
            logger.info("Notification system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize notification system: {e}")
            raise
    
    async def send_notification(self, notification: Notification) -> bool:
        """Send notification through appropriate channels."""
        try:
            # Check if notification type is enabled
            if notification.message_type not in self.enabled_types:
                return False
            
            # Check priority threshold
            if notification.priority.value < self.min_priority.value:
                return False
            
            # Check throttling
            if self._is_throttled(notification):
                logger.debug(f"Notification throttled: {notification.title}")
                return False
            
            # Store in history
            self.notification_history.append(notification)
            
            # Send via Telegram
            success = await self.telegram.send_message(notification)
            
            if success:
                self.last_sent[notification.message_type] = datetime.utcnow()
                logger.info(f"Notification sent: {notification.title}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False
    
    async def send_trade_notification(self, symbol: str, side: str, quantity: float, 
                                    price: float, confidence: float) -> bool:
        """Send trade execution notification."""
        notification = Notification(
            message_type=NotificationType.TRADE_EXECUTED,
            priority=Priority.MEDIUM,
            title=f"Trade Executed: {symbol}",
            message=f"Executed {side} order for {symbol}",
            timestamp=datetime.utcnow(),
            data={
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'value': quantity * price,
                'confidence': confidence
            }
        )
        
        return await self.send_notification(notification)
    
    async def send_performance_alert(self, alert_type: str, current_value: float, 
                                   threshold: float, message: str) -> bool:
        """Send performance alert notification."""
        priority = Priority.HIGH if abs(current_value) > abs(threshold) * 1.5 else Priority.MEDIUM
        
        notification = Notification(
            message_type=NotificationType.PERFORMANCE_ALERT,
            priority=priority,
            title=f"Performance Alert: {alert_type}",
            message=message,
            timestamp=datetime.utcnow(),
            data={
                'alert_type': alert_type,
                'current_value': current_value,
                'threshold': threshold
            }
        )
        
        return await self.send_notification(notification)
    
    async def send_risk_alert(self, risk_type: str, symbol: str, current_value: float, 
                            threshold: float, message: str) -> bool:
        """Send risk management alert."""
        notification = Notification(
            message_type=NotificationType.RISK_ALERT,
            priority=Priority.HIGH,
            title=f"Risk Alert: {risk_type}",
            message=message,
            timestamp=datetime.utcnow(),
            data={
                'risk_type': risk_type,
                'symbol': symbol,
                'current_value': current_value,
                'threshold': threshold
            }
        )
        
        return await self.send_notification(notification)
    
    async def send_system_alert(self, component: str, status: str, message: str) -> bool:
        """Send system health alert."""
        priority = Priority.CRITICAL if status == "CRITICAL" else Priority.HIGH
        
        notification = Notification(
            message_type=NotificationType.SYSTEM_ALERT,
            priority=priority,
            title=f"System Alert: {component}",
            message=message,
            timestamp=datetime.utcnow(),
            data={
                'component': component,
                'status': status
            }
        )
        
        return await self.send_notification(notification)
    
    async def send_daily_summary(self, portfolio_value: float, daily_pnl: float, 
                               total_trades: int, win_rate: float, top_performer: str) -> bool:
        """Send daily performance summary."""
        notification = Notification(
            message_type=NotificationType.DAILY_SUMMARY,
            priority=Priority.LOW,
            title="Daily Trading Summary",
            message="Here's your daily trading performance summary:",
            timestamp=datetime.utcnow(),
            data={
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl,
                'daily_return': daily_pnl / portfolio_value if portfolio_value > 0 else 0,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'top_performer': top_performer
            }
        )
        
        return await self.send_notification(notification)
    
    async def send_weekly_report(self, report_data: Dict[str, Any]) -> bool:
        """Send weekly performance report."""
        notification = Notification(
            message_type=NotificationType.WEEKLY_REPORT,
            priority=Priority.LOW,
            title="Weekly Performance Report",
            message="Your weekly trading performance report is ready:",
            timestamp=datetime.utcnow(),
            data=report_data
        )
        
        return await self.send_notification(notification)
    
    async def send_startup_notification(self) -> bool:
        """Send system startup notification."""
        notification = Notification(
            message_type=NotificationType.SYSTEM_ALERT,
            priority=Priority.MEDIUM,
            title="Crypto Pulse V3 Started",
            message="Trading system has started successfully and is monitoring markets.",
            timestamp=datetime.utcnow(),
            data={
                'version': 'V3',
                'environment': settings.trading.environment,
                'monitored_pairs': len(settings.trading.trading_pairs)
            }
        )
        
        return await self.send_notification(notification)
    
    async def send_shutdown_notification(self, reason: str = "Manual shutdown") -> bool:
        """Send system shutdown notification."""
        notification = Notification(
            message_type=NotificationType.SYSTEM_ALERT,
            priority=Priority.MEDIUM,
            title="Crypto Pulse V3 Shutdown",
            message=f"Trading system is shutting down: {reason}",
            timestamp=datetime.utcnow(),
            data={'reason': reason}
        )
        
        return await self.send_notification(notification)
    
    def _is_throttled(self, notification: Notification) -> bool:
        """Check if notification should be throttled."""
        if notification.message_type not in self.throttle_rules:
            return False
        
        if notification.priority == Priority.CRITICAL:
            return False  # Never throttle critical messages
        
        last_sent = self.last_sent.get(notification.message_type)
        if not last_sent:
            return False
        
        throttle_period = self.throttle_rules[notification.message_type]
        return datetime.utcnow() - last_sent < throttle_period
    
    def enable_notification_type(self, notification_type: NotificationType):
        """Enable a specific notification type."""
        self.enabled_types.add(notification_type)
    
    def disable_notification_type(self, notification_type: NotificationType):
        """Disable a specific notification type."""
        self.enabled_types.discard(notification_type)
    
    def set_minimum_priority(self, priority: Priority):
        """Set minimum priority for notifications."""
        self.min_priority = priority
    
    def get_notification_history(self, hours: int = 24) -> List[Notification]:
        """Get notification history for specified hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [n for n in self.notification_history if n.timestamp > cutoff]
    
    def cleanup_old_notifications(self, hours: int = 168):  # 1 week
        """Clean up old notifications from history."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        self.notification_history = [n for n in self.notification_history if n.timestamp > cutoff]


# Global notification manager instance
notification_manager = NotificationManager()
