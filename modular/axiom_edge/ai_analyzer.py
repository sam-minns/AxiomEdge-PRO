# =============================================================================
# GEMINI AI ANALYZER & API TIMER MODULE
# =============================================================================

import os
import json
import time
import requests
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class APITimer:
    """Tracks API usage and timing for rate limiting and cost management"""
    
    def __init__(self):
        self.call_log: List[Dict[str, Any]] = []
        self.total_calls = 0
        self.total_cost_estimate = 0.0
        
    def log_call(self, model: str, input_tokens: int, output_tokens: int, 
                 duration: float, cost_estimate: float = 0.0):
        """Log an API call with timing and token usage"""
        call_data = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "duration_seconds": duration,
            "cost_estimate": cost_estimate
        }
        self.call_log.append(call_data)
        self.total_calls += 1
        self.total_cost_estimate += cost_estimate
        
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of API usage"""
        if not self.call_log:
            return {"total_calls": 0, "total_cost": 0.0}
            
        total_tokens = sum(call["total_tokens"] for call in self.call_log)
        avg_duration = sum(call["duration_seconds"] for call in self.call_log) / len(self.call_log)
        
        return {
            "total_calls": self.total_calls,
            "total_tokens": total_tokens,
            "total_cost_estimate": self.total_cost_estimate,
            "average_duration": avg_duration,
            "calls_last_hour": len([c for c in self.call_log 
                                  if (datetime.now() - datetime.fromisoformat(c["timestamp"])).seconds < 3600])
        }


class GeminiAnalyzer:
    """
    AI-powered analysis using Google's Gemini API.
    Can be used independently for AI-driven insights and recommendations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_key_valid = True
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model_priority_list = [
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest", 
            "gemini-pro"
        ]
        
        # Tool configuration for web search capabilities
        self.tools = [{
            "function_declarations": [{
                "name": "search_web",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }]
        }]
        
        self.tool_config = {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["search_web"]
            }
        }
        
        if not self._validate_api_key():
            self._handle_missing_api_key()
    
    def _validate_api_key(self) -> bool:
        """Validate the API key"""
        if not self.api_key or "YOUR" in self.api_key or "PASTE" in self.api_key:
            return False
        return True
    
    def _handle_missing_api_key(self):
        """Handle missing or invalid API key"""
        logger.warning("!CRITICAL! GEMINI_API_KEY not found in environment or is a placeholder.")
        try:
            self.api_key = input(">>> Please paste your Gemini API Key and press Enter, or press Enter to skip: ").strip()
            if not self.api_key:
                logger.warning("No API Key provided. AI analysis will be skipped.")
                self.api_key_valid = False
            else:
                logger.info("Using API Key provided via manual input.")
        except Exception:
            logger.warning("Could not read input (non-interactive environment?). AI analysis will be skipped.")
            self.api_key_valid = False
            self.api_key = None

    def _call_gemini(self, prompt: str, model: Optional[str] = None) -> str:
        """Make a call to the Gemini API"""
        if not self.api_key_valid:
            logger.warning("Gemini API key is not valid. Skipping API call.")
            return "{}"

        max_prompt_length = 950000
        if len(prompt) > max_prompt_length:
            logger.warning(f"Prompt length ({len(prompt)}) is very large, approaching the model's context window limit.")

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": self.tools,
            "tool_config": self.tool_config
        }
        
        # Remove tools if not needed
        if "search_web" not in prompt:
            payload.pop("tools", None)
            payload.pop("tool_config", None)

        models_to_try = [model] if model else self.model_priority_list
        retry_delays = [5, 15, 30]  # Seconds

        for model_name in models_to_try:
            for attempt, delay in enumerate(retry_delays):
                try:
                    start_time = time.time()
                    url = f"{self.base_url}/{model_name}:generateContent?key={self.api_key}"
                    
                    response = requests.post(url, json=payload, timeout=120)
                    duration = time.time() - start_time
                    
                    if response.status_code == 200:
                        result = response.json()
                        if "candidates" in result and result["candidates"]:
                            content = result["candidates"][0]["content"]["parts"][0]["text"]
                            logger.info(f"✓ Gemini API call successful with {model_name} (attempt {attempt + 1})")
                            return content
                        else:
                            logger.warning(f"Empty response from {model_name}")
                            
                    elif response.status_code == 429:
                        logger.warning(f"Rate limit hit for {model_name}, waiting {delay}s...")
                        time.sleep(delay)
                        continue
                        
                    else:
                        logger.error(f"API error {response.status_code}: {response.text}")
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout for {model_name} (attempt {attempt + 1})")
                except Exception as e:
                    logger.error(f"Error calling {model_name}: {e}")
                
                if attempt < len(retry_delays) - 1:
                    time.sleep(delay)
            
            logger.warning(f"All attempts failed for {model_name}, trying next model...")
        
        logger.error("All Gemini models failed. Returning empty response.")
        return "{}"

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from AI response text"""
        try:
            # Try to find JSON in code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            else:
                # Try to find JSON-like structure
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to extract JSON from response: {e}")
            return {}

    def _sanitize_dict(self, data: Any) -> Any:
        """Recursively sanitize dictionary for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._sanitize_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_dict(item) for item in data]
        elif isinstance(data, (pd.Timestamp, datetime)):
            return data.isoformat()
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return str(data)
        elif hasattr(data, '__dict__'):
            return str(data)
        else:
            return data

    def analyze_market_regime(self, market_data: pd.DataFrame, symbols: List[str]) -> Dict[str, Any]:
        """Analyze current market regime using AI"""
        if not self.api_key_valid:
            return {"regime": "unknown", "confidence": 0.0, "analysis": "AI analysis skipped"}
        
        # Prepare market data summary
        data_summary = self._prepare_market_summary(market_data, symbols)
        
        prompt = f"""
        Analyze the current market regime based on the following data:
        
        Market Data Summary:
        {data_summary}
        
        Please analyze and classify the current market regime into one of these categories:
        - TRENDING_BULL: Strong upward trend with momentum
        - TRENDING_BEAR: Strong downward trend with momentum  
        - SIDEWAYS_CONSOLIDATION: Range-bound, low volatility
        - HIGH_VOLATILITY: High volatility, uncertain direction
        - CRISIS_MODE: Extreme volatility, potential crisis conditions
        
        Return your analysis in JSON format:
        {{
            "regime": "REGIME_NAME",
            "confidence": 0.85,
            "key_indicators": ["indicator1", "indicator2"],
            "analysis": "Detailed explanation of the regime classification",
            "recommended_strategy": "Strategy recommendation for this regime"
        }}
        """
        
        response = self._call_gemini(prompt)
        return self._extract_json_from_response(response)

    def _prepare_market_summary(self, market_data: pd.DataFrame, symbols: List[str]) -> str:
        """Prepare a summary of market data for AI analysis"""
        summary_parts = []
        
        for symbol in symbols[:5]:  # Limit to first 5 symbols
            symbol_data = market_data[market_data['symbol'] == symbol].tail(50)
            if not symbol_data.empty:
                recent_return = (symbol_data['Close'].iloc[-1] / symbol_data['Close'].iloc[0] - 1) * 100
                volatility = symbol_data['Close'].pct_change().std() * 100
                
                summary_parts.append(f"{symbol}: {recent_return:.2f}% return, {volatility:.2f}% volatility")
        
        return "\n".join(summary_parts)

    def get_broker_analysis(self, symbol: str, broker_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze broker-specific information using AI with web search"""
        if not self.api_key_valid:
            return {"analysis": "AI analysis skipped", "recommendations": []}
        
        prompt = f"""
        search_web("current {symbol} spread analysis broker comparison 2024")
        
        Analyze the broker information for {symbol} and provide insights:
        
        Current Broker Data:
        {json.dumps(broker_data, indent=2)}
        
        Please provide analysis in JSON format:
        {{
            "spread_analysis": "Analysis of current spreads",
            "cost_efficiency": "Cost efficiency rating",
            "recommendations": ["recommendation1", "recommendation2"],
            "market_conditions": "Current market conditions affecting spreads"
        }}
        """
        
        response = self._call_gemini(prompt)
        return self._extract_json_from_response(response)

    def propose_strategy_optimization(self, performance_data: Dict[str, Any], 
                                    market_regime: str) -> Dict[str, Any]:
        """Propose strategy optimizations based on performance and market conditions"""
        if not self.api_key_valid:
            return {}
        
        sanitized_performance = self._sanitize_dict(performance_data)
        
        prompt = f"""
        Based on the following performance data and market regime, propose strategy optimizations:
        
        Performance Data:
        {json.dumps(sanitized_performance, indent=2)}
        
        Current Market Regime: {market_regime}
        
        Please provide optimization recommendations in JSON format:
        {{
            "parameter_adjustments": {{
                "risk_per_trade": 0.02,
                "stop_loss_pct": 0.05
            }},
            "feature_recommendations": ["feature1", "feature2"],
            "strategy_modifications": "Detailed strategy modification suggestions",
            "risk_management": "Risk management recommendations",
            "expected_improvement": "Expected performance improvement"
        }}
        """
        
        response = self._call_gemini(prompt)
        return self._extract_json_from_response(response)
