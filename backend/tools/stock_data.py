"""
Stock Data Tool Module
Provides stock market data fetching with Finnhub as primary source and yfinance as fallback.
Includes caching layer to minimize API calls.
"""

import yfinance as yf
import finnhub
import requests
import time
import json
import os
import logging
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

# API Configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

if not FINNHUB_API_KEY:
    raise ValueError(
        "FINNHUB_API_KEY not found in environment variables. "
        f"Checked: {ENV_PATH}"
    )

# HTTP Session Configuration
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

# Finnhub Client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Cache Configuration
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================
# STOCK DATA TOOL
# ==============================

class StockDataTool:
    """
    Tool for fetching stock market data with multi-source support and caching.
    
    Primary data source: Finnhub API
    Fallback data source: yfinance
    Caching: JSON file-based cache to reduce API calls
    """
    
    def __init__(self):
        """Initialize the stock data tool."""
        self.name = "stock_data_fetcher"
        self.description = "Fetch stock data using Finnhub with yfinance fallback"
        
    # ==============================
    # CACHE MANAGEMENT
    # ==============================
    
    def get_cache_path(self, ticker: str) -> Path:
        """
        Get the cache file path for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Path to the cache file
        """
        return CACHE_DIR / f"{ticker}.json"
    
    def load_cache(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Load cached data for a ticker if available.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Cached data dict or None if not found
        """
        cache_path = self.get_cache_path(ticker)
        
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    logger.info(f"Cache hit for {ticker}")
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache for {ticker}: {e}")
                
        return None
    
    def save_cache(self, ticker: str, data: Dict[str, Any]) -> None:
        """
        Save data to cache for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            data: Data to cache
        """
        cache_path = self.get_cache_path(ticker)
        
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Cached data for {ticker}")
        except IOError as e:
            logger.error(f"Failed to save cache for {ticker}: {e}")
    
    # ==============================
    # DATA SOURCE: FINNHUB
    # ==============================
    
    def get_from_finnhub(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch stock data from Finnhub API.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Stock data dict or None if fetch fails
        """
        try:
            logger.info(f"Fetching {ticker} from Finnhub API")
            quote = finnhub_client.quote(ticker)
            
            if quote and quote.get("c") != 0:
                return {
                    "ticker": ticker,
                    "current_price": quote["c"],
                    "high": quote["h"],
                    "low": quote["l"],
                    "open": quote["o"],
                    "prev_close": quote["pc"],
                    "source": "finnhub"
                }
                
        except Exception as e:
            logger.warning(f"Finnhub fetch failed for {ticker}: {e}")
            
        return None
    
    # ==============================
    # DATA SOURCE: YFINANCE
    # ==============================
    
    def get_from_yfinance(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch stock data from yfinance (fallback source).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Stock data dict or None if fetch fails
        """
        try:
            logger.info(f"Fetching {ticker} from yfinance (fallback)")
            time.sleep(1)  # Rate limiting
            
            stock = yf.Ticker(ticker, session=session)
            hist = stock.history(period="1d")
            
            if not hist.empty:
                return {
                    "ticker": ticker,
                    "current_price": float(hist['Close'].iloc[-1]),
                    "high": float(hist['High'].iloc[-1]),
                    "low": float(hist['Low'].iloc[-1]),
                    "open": float(hist['Open'].iloc[-1]),
                    "source": "yfinance"
                }
                
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {ticker}: {e}")
            
        return None
    
    # ==============================
    # PUBLIC INTERFACE
    # ==============================
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get stock information with automatic source failover and caching.
        
        Data retrieval strategy:
        1. Check cache first
        2. Try Finnhub API (primary)
        3. Fallback to yfinance
        4. Return error if all sources fail
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            
        Returns:
            Dictionary containing stock data or error message
            
        Example:
            >>> tool = StockDataTool()
            >>> data = tool.get_stock_info("AAPL")
            >>> print(data["current_price"])
            258.86
        """
        ticker = ticker.upper().strip()
        
        # Step 1: Check cache
        cached_data = self.load_cache(ticker)
        if cached_data:
            cached_data["source"] = "cache"
            return cached_data
        
        # Step 2: Try Finnhub (primary source)
        data = self.get_from_finnhub(ticker)
        if data:
            self.save_cache(ticker, data)
            return data
        
        # Step 3: Fallback to yfinance
        data = self.get_from_yfinance(ticker)
        if data:
            self.save_cache(ticker, data)
            return data
        
        # Step 4: All sources failed
        error_msg = f"Unable to fetch data for {ticker} from any source"
        logger.error(error_msg)
        return {
            "ticker": ticker,
            "error": error_msg,
            "source": "none"
        }
    
    def get_quarterly_revenue(
        self, 
        ticker: str, 
        year: int, 
        quarter: int
    ) -> Dict[str, Any]:
        """
        Fetch quarterly revenue data for a company.
        
        Args:
            ticker: Stock ticker symbol
            year: Year (e.g., 2025)
            quarter: Quarter number (1-4)
            
        Returns:
            Dictionary containing quarterly revenue data
        """
        try:
            logger.info(f"Fetching Q{quarter} {year} revenue for {ticker}")
            
            stock = yf.Ticker(ticker, session=session)
            quarterly_financials = stock.quarterly_financials
            
            if quarterly_financials.empty:
                logger.warning(f"No financial data available for {ticker}")
                return {
                    "ticker": ticker,
                    "error": "No quarterly financial data available"
                }
            
            # Get total revenue row
            if 'Total Revenue' in quarterly_financials.index:
                revenue_row = quarterly_financials.loc['Total Revenue']
                
                return {
                    "ticker": ticker,
                    "year": year,
                    "quarter": quarter,
                    "latest_quarter_revenue": float(revenue_row.iloc[0]),
                    "currency": "USD",
                    "data": revenue_row.to_dict(),
                    "source": "yfinance"
                }
            else:
                return {
                    "ticker": ticker,
                    "error": "Revenue data not found in financial statements"
                }
                
        except Exception as e:
            logger.error(f"Failed to fetch quarterly revenue for {ticker}: {e}")
            return {
                "ticker": ticker,
                "error": str(e)
            }
    
    def clear_cache(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear cache for a specific ticker or all tickers.
        
        Args:
            ticker: Specific ticker to clear, or None to clear all
            
        Returns:
            Status dictionary
        """
        try:
            if ticker:
                cache_path = self.get_cache_path(ticker.upper())
                if cache_path.exists():
                    cache_path.unlink()
                    logger.info(f"Cleared cache for {ticker}")
                    return {"status": "success", "message": f"Cache cleared for {ticker}"}
                else:
                    return {"status": "info", "message": f"No cache found for {ticker}"}
            else:
                # Clear all cache files
                cache_files = list(CACHE_DIR.glob("*.json"))
                for cache_file in cache_files:
                    cache_file.unlink()
                logger.info(f"Cleared all cache ({len(cache_files)} files)")
                return {
                    "status": "success", 
                    "message": f"Cleared {len(cache_files)} cache files"
                }
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return {"status": "error", "message": str(e)}


# ==============================
# MODULE EXPORT
# ==============================

__all__ = ['StockDataTool']