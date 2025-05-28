"""
Sentiment analysis system integrating multiple data sources.
Primary integration with Perplexity AI for news and social sentiment.
"""

import asyncio
import aiohttp
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from textblob import TextBlob
import numpy as np

from src.core.database import db_manager
from src.core.models import SentimentData
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Container for sentiment analysis results."""
    symbol: Optional[str]
    timestamp: datetime
    news_sentiment: float  # -1 to 1
    social_sentiment: float  # -1 to 1
    fear_greed_index: float  # 0 to 100
    overall_sentiment: float  # Weighted average
    confidence_score: float  # 0 to 1
    trend: str  # 'bullish', 'bearish', 'neutral'
    key_events: List[str]
    sentiment_drivers: Dict[str, float]


@dataclass
class NewsItem:
    """Container for news item data."""
    title: str
    content: str
    source: str
    timestamp: datetime
    relevance_score: float
    sentiment_score: float


class PerplexityAIClient:
    """Client for Perplexity AI API integration."""
    
    def __init__(self):
        self.api_key = settings.perplexity_api_key
        self.base_url = "https://api.perplexity.ai"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self.session
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_crypto_news_sentiment(self, symbol: str = None) -> Dict:
        """
        Get cryptocurrency news and sentiment from Perplexity AI.
        
        Args:
            symbol: Specific cryptocurrency symbol (optional)
            
        Returns:
            Dictionary containing news and sentiment analysis
        """
        try:
            session = await self._get_session()
            
            # Construct query based on symbol
            if symbol:
                # Remove 'USDT' or other quote currencies for cleaner search
                clean_symbol = symbol.replace('USDT', '').replace('USD', '').replace('BTC', '')
                query = f"Latest news and market sentiment for {clean_symbol} cryptocurrency in the last 24 hours. Include price movements, institutional adoption, regulatory news, and social media sentiment."
            else:
                query = "Latest cryptocurrency market news and overall sentiment in the last 24 hours. Include Bitcoin, Ethereum, major altcoins, institutional adoption, regulatory developments, and market sentiment indicators."
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency market analyst. Provide comprehensive analysis of recent news and sentiment, including specific events, price catalysts, and market sentiment indicators. Structure your response with clear sections for news events, sentiment analysis, and market implications."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3,
                "top_p": 0.9,
                "return_citations": True,
                "search_domain_filter": ["perplexity.ai"],
                "return_images": False,
                "return_related_questions": False,
                "search_recency_filter": "day"
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_perplexity_response(data, symbol)
                else:
                    logger.error(f"Perplexity API error: {response.status}")
                    return self._empty_sentiment_response(symbol)
                    
        except Exception as e:
            logger.error(f"Error getting sentiment from Perplexity AI: {e}")
            return self._empty_sentiment_response(symbol)
    
    def _parse_perplexity_response(self, response_data: Dict, symbol: str = None) -> Dict:
        """Parse Perplexity AI response and extract sentiment information."""
        try:
            content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
            citations = response_data.get('citations', [])
            
            # Extract sentiment using keyword analysis and TextBlob
            sentiment_keywords = {
                'bullish': ['bullish', 'positive', 'gains', 'rally', 'surge', 'upward', 'growth', 'adoption', 'institutional'],
                'bearish': ['bearish', 'negative', 'losses', 'decline', 'crash', 'fall', 'regulatory concerns', 'ban'],
                'neutral': ['stable', 'consolidation', 'sideways', 'unchanged', 'mixed signals']
            }
            
            # Count sentiment keywords
            content_lower = content.lower()
            sentiment_scores = {}
            
            for sentiment, keywords in sentiment_keywords.items():
                score = sum(1 for keyword in keywords if keyword in content_lower)
                sentiment_scores[sentiment] = score
            
            # Calculate overall sentiment score (-1 to 1)
            total_keywords = sum(sentiment_scores.values())
            if total_keywords > 0:
                bullish_ratio = sentiment_scores['bullish'] / total_keywords
                bearish_ratio = sentiment_scores['bearish'] / total_keywords
                overall_sentiment = bullish_ratio - bearish_ratio
            else:
                # Fallback to TextBlob sentiment
                blob = TextBlob(content)
                overall_sentiment = blob.sentiment.polarity
            
            # Extract key events from content
            key_events = self._extract_key_events(content)
            
            # Determine trend
            if overall_sentiment > 0.1:
                trend = 'bullish'
            elif overall_sentiment < -0.1:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Calculate confidence based on content length and keyword density
            confidence = min(1.0, len(content) / 1000 + total_keywords / 20)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'news_sentiment': overall_sentiment,
                'social_sentiment': overall_sentiment * 0.8,  # Slightly discounted
                'overall_sentiment': overall_sentiment,
                'confidence_score': confidence,
                'trend': trend,
                'key_events': key_events,
                'raw_content': content,
                'citations': citations,
                'sentiment_scores': sentiment_scores
            }
            
        except Exception as e:
            logger.error(f"Error parsing Perplexity response: {e}")
            return self._empty_sentiment_response(symbol)
    
    def _extract_key_events(self, content: str) -> List[str]:
        """Extract key events from news content."""
        try:
            # Simple extraction based on sentence patterns
            sentences = content.split('.')
            key_events = []
            
            # Look for sentences with key indicators
            event_indicators = [
                'announced', 'launched', 'approved', 'rejected', 'filed',
                'partnership', 'acquisition', 'regulatory', 'adoption',
                'institutional', 'whale', 'major', 'significant'
            ]
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(indicator in sentence.lower() for indicator in event_indicators):
                    if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length
                        key_events.append(sentence)
            
            return key_events[:5]  # Limit to top 5 events
            
        except Exception as e:
            logger.error(f"Error extracting key events: {e}")
            return []
    
    def _empty_sentiment_response(self, symbol: str = None) -> Dict:
        """Return empty sentiment response for error cases."""
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow(),
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'overall_sentiment': 0.0,
            'confidence_score': 0.0,
            'trend': 'neutral',
            'key_events': [],
            'raw_content': '',
            'citations': [],
            'sentiment_scores': {}
        }


class FearGreedIndexClient:
    """Client for Fear & Greed Index data."""
    
    def __init__(self):
        self.api_url = "https://api.alternative.me/fng/"
    
    async def get_fear_greed_index(self) -> float:
        """Get current Fear & Greed Index value."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract current value
                        current_data = data.get('data', [{}])[0]
                        value = float(current_data.get('value', 50))
                        
                        # Normalize to -1 to 1 scale
                        # 0-25: Extreme Fear (-1 to -0.5)
                        # 25-45: Fear (-0.5 to -0.1)
                        # 45-55: Neutral (-0.1 to 0.1)
                        # 55-75: Greed (0.1 to 0.5)
                        # 75-100: Extreme Greed (0.5 to 1)
                        
                        if value <= 25:
                            normalized = -1 + (value / 25) * 0.5
                        elif value <= 45:
                            normalized = -0.5 + ((value - 25) / 20) * 0.4
                        elif value <= 55:
                            normalized = -0.1 + ((value - 45) / 10) * 0.2
                        elif value <= 75:
                            normalized = 0.1 + ((value - 55) / 20) * 0.4
                        else:
                            normalized = 0.5 + ((value - 75) / 25) * 0.5
                        
                        return normalized
                    
        except Exception as e:
            logger.error(f"Error getting Fear & Greed Index: {e}")
            
        return 0.0  # Neutral fallback


class SentimentAnalyzer:
    """
    Main sentiment analysis coordinator.
    Integrates multiple sentiment sources and generates trading signals.
    """
    
    def __init__(self):
        self.perplexity_client = PerplexityAIClient()
        self.fear_greed_client = FearGreedIndexClient()
        self.sentiment_cache = {}  # Cache for recent sentiment data
        self.cache_duration = timedelta(hours=1)  # Cache duration
    
    async def analyze_market_sentiment(self, symbol: str = None) -> SentimentScore:
        """
        Analyze overall market sentiment for a symbol or general market.
        
        Args:
            symbol: Specific cryptocurrency symbol (optional)
            
        Returns:
            SentimentScore object with comprehensive sentiment analysis
        """
        try:
            # Check cache first
            cache_key = symbol or 'general'
            if cache_key in self.sentiment_cache:
                cached_data, cached_time = self.sentiment_cache[cache_key]
                if datetime.utcnow() - cached_time < self.cache_duration:
                    return cached_data
            
            # Get sentiment from multiple sources
            perplexity_data = await self.perplexity_client.get_crypto_news_sentiment(symbol)
            fear_greed_score = await self.fear_greed_client.get_fear_greed_index()
            
            # Historical sentiment trend
            historical_sentiment = await self._get_historical_sentiment_trend(symbol)
            
            # Combine sentiment scores with weights
            news_sentiment = perplexity_data['news_sentiment']
            social_sentiment = perplexity_data['social_sentiment']
            
            # Calculate overall sentiment (weighted average)
            sentiment_weights = {
                'news': 0.4,
                'social': 0.3,
                'fear_greed': 0.2,
                'historical': 0.1
            }
            
            overall_sentiment = (
                sentiment_weights['news'] * news_sentiment +
                sentiment_weights['social'] * social_sentiment +
                sentiment_weights['fear_greed'] * fear_greed_score +
                sentiment_weights['historical'] * historical_sentiment
            )
            
            # Determine trend
            if overall_sentiment > 0.15:
                trend = 'bullish'
            elif overall_sentiment < -0.15:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Calculate confidence score
            confidence = min(1.0, perplexity_data['confidence_score'] + 0.2)
            
            # Identify sentiment drivers
            sentiment_drivers = {
                'news_impact': abs(news_sentiment),
                'social_impact': abs(social_sentiment),
                'fear_greed_impact': abs(fear_greed_score),
                'trend_strength': abs(overall_sentiment)
            }
            
            sentiment_score = SentimentScore(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                fear_greed_index=fear_greed_score,
                overall_sentiment=overall_sentiment,
                confidence_score=confidence,
                trend=trend,
                key_events=perplexity_data['key_events'],
                sentiment_drivers=sentiment_drivers
            )
            
            # Cache the result
            self.sentiment_cache[cache_key] = (sentiment_score, datetime.utcnow())
            
            # Store to database
            await self._store_sentiment_data(sentiment_score, perplexity_data)
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return self._empty_sentiment_score(symbol)
    
    async def _get_historical_sentiment_trend(self, symbol: str = None) -> float:
        """Get historical sentiment trend for context."""
        try:
            with db_manager.get_session() as session:
                # Get sentiment data from last 7 days
                start_date = datetime.utcnow() - timedelta(days=7)
                
                query = session.query(SentimentData).filter(
                    SentimentData.timestamp >= start_date
                )
                
                if symbol:
                    query = query.filter(SentimentData.symbol == symbol)
                else:
                    query = query.filter(SentimentData.symbol.is_(None))
                
                sentiment_records = query.order_by(SentimentData.timestamp.desc()).limit(20).all()
                
                if not sentiment_records:
                    return 0.0
                
                # Calculate trend (recent vs older sentiment)
                recent_sentiment = np.mean([
                    float(r.news_sentiment or 0) + float(r.social_sentiment or 0)
                    for r in sentiment_records[:5]
                ])
                
                older_sentiment = np.mean([
                    float(r.news_sentiment or 0) + float(r.social_sentiment or 0)
                    for r in sentiment_records[5:10]
                ]) if len(sentiment_records) > 5 else recent_sentiment
                
                # Return normalized trend
                trend = (recent_sentiment - older_sentiment) / 2  # Normalize
                return max(-1.0, min(1.0, trend))
                
        except Exception as e:
            logger.error(f"Error getting historical sentiment trend: {e}")
            return 0.0
    
    async def _store_sentiment_data(self, sentiment_score: SentimentScore, raw_data: Dict):
        """Store sentiment data to database."""
        try:
            with db_manager.get_session() as session:
                sentiment_data = SentimentData(
                    timestamp=sentiment_score.timestamp,
                    symbol=sentiment_score.symbol,
                    news_sentiment=sentiment_score.news_sentiment,
                    social_sentiment=sentiment_score.social_sentiment,
                    fear_greed_index=sentiment_score.fear_greed_index,
                    source='perplexity_ai',
                    raw_data=raw_data,
                    sentiment_trend=sentiment_score.overall_sentiment,
                    confidence_score=sentiment_score.confidence_score
                )
                
                session.add(sentiment_data)
                session.commit()
                
                logger.debug(f"Stored sentiment data for {sentiment_score.symbol or 'general'}")
                
        except Exception as e:
            logger.error(f"Error storing sentiment data: {e}")
    
    def _empty_sentiment_score(self, symbol: str = None) -> SentimentScore:
        """Return empty sentiment score for error cases."""
        return SentimentScore(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            news_sentiment=0.0,
            social_sentiment=0.0,
            fear_greed_index=0.0,
            overall_sentiment=0.0,
            confidence_score=0.0,
            trend='neutral',
            key_events=[],
            sentiment_drivers={}
        )
    
    async def get_sentiment_signal_strength(self, symbol: str = None) -> float:
        """
        Get sentiment signal strength for trading decisions.
        
        Returns:
            Signal strength from -1.0 to 1.0
            - Positive values suggest bullish sentiment
            - Negative values suggest bearish sentiment
        """
        try:
            sentiment_score = await self.analyze_market_sentiment(symbol)
            
            # Weight by confidence
            signal_strength = sentiment_score.overall_sentiment * sentiment_score.confidence_score
            
            # Apply momentum factor based on trend consistency
            if sentiment_score.trend != 'neutral':
                momentum_factor = 1.2  # Boost signal when trend is clear
                signal_strength *= momentum_factor
            
            return max(-1.0, min(1.0, signal_strength))
            
        except Exception as e:
            logger.error(f"Error getting sentiment signal strength: {e}")
            return 0.0
    
    async def analyze_symbol_sentiment(self, symbol: str) -> SentimentScore:
        """Analyze sentiment for a specific symbol."""
        return await self.analyze_market_sentiment(symbol)
    
    async def update_fear_greed_index(self):
        """Update the fear and greed index from external sources."""
        try:
            fear_greed_value = await self.fear_greed_client.get_fear_greed_index()
            logger.info(f"Updated fear & greed index: {fear_greed_value}")
        except Exception as e:
            logger.error(f"Error updating fear & greed index: {e}")
    
    async def get_symbol_sentiment(self, symbol: str) -> float:
        """Get sentiment score for a specific symbol (-1 to 1)."""
        try:
            sentiment_score = await self.analyze_market_sentiment(symbol)
            return sentiment_score.overall_sentiment
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return 0.0
    
    async def close(self):
        """Close all HTTP sessions."""
        await self.perplexity_client.close()


# Global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer()
