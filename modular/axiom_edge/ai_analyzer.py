# =============================================================================
# GEMINI AI ANALYZER & API TIMER MODULE
# =============================================================================

import os
import re
import json
import time
import sys
import requests
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta, date
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)

def flush_loggers():
    """
    Flush all handlers for all active loggers to disk.

    Ensures all pending log messages are written to their respective outputs
    before the application continues or terminates.
    """
    for handler in logging.getLogger().handlers:
        handler.flush()
    for handler in logging.getLogger("ML_Trading_Framework").handlers:
        handler.flush()

class APITimer:
    """
    Rate limiting timer for API calls to prevent exceeding quotas.

    Ensures minimum intervals between API calls to comply with rate limits
    and prevent service disruption. Provides automatic waiting and call tracking.

    Attributes:
        interval: Minimum time between API calls
        last_call_time: Timestamp of the last API call
    """

    def __init__(self, interval_seconds: int = 61):
        self.interval = timedelta(seconds=interval_seconds)
        self.last_call_time: Optional[datetime] = None
        if self.interval.total_seconds() > 0:
            logger.info(f"API Timer initialized with a {self.interval.total_seconds():.0f}-second interval.")
        else:
            logger.info("API Timer initialized with a 0-second interval (timer is effectively disabled).")

    def _wait_if_needed(self):
        if self.interval.total_seconds() <= 0:
            return
        if self.last_call_time is None:
            return
        elapsed = datetime.now() - self.last_call_time
        wait_time_delta = self.interval - elapsed
        wait_seconds = wait_time_delta.total_seconds()
        if wait_seconds > 0:
            logger.info(f"  - Time since last API call: {elapsed.total_seconds():.1f} seconds.")
            logger.info(f"  - Waiting for {wait_seconds:.1f} seconds to respect the {self.interval.total_seconds():.0f}s interval...")
            if hasattr(logging, 'flush') and callable(logging.flush):
                logging.flush()
            elif sys.stdout and hasattr(sys.stdout, 'flush') and callable(sys.stdout.flush):
                sys.stdout.flush()
            time.sleep(wait_seconds)
        else:
            logger.info(f"  - Time since last API call ({elapsed.total_seconds():.1f}s) exceeds interval. No wait needed.")

    def call(self, api_function: Callable, *args, **kwargs) -> Any:
        self._wait_if_needed()
        self.last_call_time = datetime.now()
        logger.info(f"  - Making API call to '{api_function.__name__}' at {self.last_call_time.strftime('%H:%M:%S')}...")
        result = api_function(*args, **kwargs)
        logger.info(f"  - API call to '{api_function.__name__}' complete.")
        return result


class GeminiAnalyzer:
    """
    AI-powered analysis engine using Google's Gemini API.

    Provides intelligent analysis for trading strategy optimization, parameter
    tuning, market condition assessment, and asset classification. Handles API
    authentication, rate limiting, and response parsing with fallback mechanisms.

    Features:
    - Automatic API key detection and validation
    - Interactive key input for missing credentials
    - Intelligent prompt engineering for trading analysis
    - JSON response parsing with error handling
    - Asset classification and strategy recommendations

    Attributes:
        api_key: Gemini API key from environment or user input
        api_key_valid: Flag indicating if API key is available and valid
        headers: HTTP headers for API requests
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_key_valid = True  # Assume valid initially
        if not self.api_key or "YOUR" in self.api_key or "PASTE" in self.api_key:
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
        else:
            logger.info("Successfully loaded GEMINI_API_KEY from environment.")

        self.headers = {"Content-Type": "application/json"}
        # Replace single model attributes with a prioritized list.
        self.model_priority_list = [
            "gemini-2.5-flash-lite-preview-06-17",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite"
        ]
        logger.info(f"AI model priority set to: {self.model_priority_list}")
        self.tools = [{"function_declarations": [{"name": "search_web", "description": "Searches web.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]}]
        self.tool_config = {"function_calling_config": {"mode": "AUTO"}}

    def _sanitize_dict(self, data: Any) -> Any:
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                if isinstance(k, (datetime, date, pd.Timestamp)):
                    new_key = k.isoformat()
                elif not isinstance(k, (str, int, float, bool, type(None))):
                    new_key = str(k)
                else:
                    new_key = k
                new_dict[new_key] = self._sanitize_dict(v)
            return new_dict

        elif isinstance(data, list):
            return [self._sanitize_dict(elem) for elem in data]
        elif isinstance(data, (datetime, date)):
            return data.isoformat()
        elif isinstance(data, Enum):
            return data.value
        elif isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
            return int(data)
        elif isinstance(data, (np.float64, np.float32, np.float16)):
            if np.isnan(data) or np.isinf(data):
                return None
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)

    def _call_gemini(self, prompt: str) -> str:
        if not self.api_key_valid:
            logger.warning("Gemini API key is not valid. Skipping API call.")
            return "{}"

        max_prompt_length = 950000

        if len(prompt) > max_prompt_length:
            logger.warning(f"Prompt length ({len(prompt)}) is very large, approaching the model's context window limit of ~1M tokens.")

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": self.tools,
            "tool_config": self.tool_config
        }

        if "search_web" not in prompt:
            payload.pop("tools", None)
            payload.pop("tool_config", None)

        sanitized_payload = self._sanitize_dict(payload)

        models_to_try = self.model_priority_list
        retry_delays = [5, 15, 30] # Seconds

        for model_name in models_to_try:
            logger.info(f"Attempting to call Gemini API with model: {model_name}")
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"

            for attempt, delay in enumerate([0] + retry_delays):
                if delay > 0:
                    logger.warning(f"Retrying API call to {model_name} in {delay} seconds... (Attempt {attempt + 1}/{len(retry_delays) + 1})")
                    flush_loggers()
                    time.sleep(delay)
                try:
                    response = requests.post(api_url, headers=self.headers, data=json.dumps(sanitized_payload), timeout=180)
                    response.raise_for_status()

                    result = response.json()

                    if "candidates" in result and result["candidates"]:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                            text_part = candidate["content"]["parts"][0].get("text")
                            if text_part:
                                cleaned_text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text_part)
                                logger.info(f"Successfully received and extracted text response from model: {model_name}")
                                return cleaned_text

                    logger.error(f"Invalid Gemini response structure from {model_name}: No 'text' part found. Response: {result}")

                except requests.exceptions.HTTPError as e:
                    logger.error(f"!! HTTP Error for model '{model_name}': {e.response.status_code} {e.response.reason}")
                    if e.response and e.response.text:
                        logger.error(f"   - API Error Details: {e.response.text}")
                    if e.response is not None and e.response.status_code in [400, 401, 403, 404, 429]:
                        break
                except requests.exceptions.RequestException as e:
                    logger.error(f"Gemini API request failed for model {model_name} on attempt {attempt + 1} (Network Error): {e}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode Gemini response JSON from {model_name}: {e} - Response Text: {response.text if 'response' in locals() else 'N/A'}")

            logger.warning(f"Failed to get a valid text response from model {model_name} after all retries.")

        logger.critical("API connection failed for all primary and backup models. Could not get a final text response.")
        return "{}"

    def _extract_json_from_response(self, response_text: str) -> dict:
        logger.debug(f"RAW AI RESPONSE TO BE PARSED:\n--- START ---\n{response_text}\n--- END ---")

        match_backticks = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
        if match_backticks:
            json_str = match_backticks.group(1).strip()
            # Remove common JSON errors like comments and trailing commas before parsing
            json_str = re.sub(r"//.*", "", json_str)
            json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
            try:
                suggestions = json.loads(json_str)
                if isinstance(suggestions, dict):
                    logger.info("Successfully extracted JSON object from within backticks.")
                    # Handle nested params if present (for some AI calls)
                    if isinstance(suggestions.get("current_params"), dict):
                        nested_params = suggestions.pop("current_params")
                        suggestions.update(nested_params)
                    return suggestions
                else:
                    logger.warning(f"Parsed JSON from backticks was type {type(suggestions)}, not a dictionary. Trying fallback.")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decoding from backticks failed: {e}. Trying fallback heuristic.")

        try:
            start_brace = response_text.find('{')
            end_brace = response_text.rfind('}')
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                json_str_heuristic = response_text[start_brace : end_brace + 1]
                # Also clean the heuristic extraction for robustness
                json_str_heuristic = re.sub(r"//.*", "", json_str_heuristic)
                json_str_heuristic = re.sub(r",(\s*[}\]])", r"\1", json_str_heuristic)
                suggestions = json.loads(json_str_heuristic)
                if isinstance(suggestions, dict):
                    logger.info("Successfully extracted JSON object using outer brace heuristic.")
                    # Handle nested params if present
                    if isinstance(suggestions.get("current_params"), dict):
                        nested_params = suggestions.pop("current_params")
                        suggestions.update(nested_params)
                    return suggestions
                else:
                    logger.warning(f"Parsed JSON from heuristic was type {type(suggestions)}, not a dictionary.")
            else:
                logger.warning("Could not find valid JSON braces in response.")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decoding from heuristic failed: {e}")

        logger.error("All JSON extraction methods failed. Returning empty dictionary.")
        return {}

    def classify_asset_symbols(self, symbols: List[str]) -> Dict[str, str]:
        """
        Classify financial symbols into asset classes using AI analysis.

        Uses the Gemini API to intelligently categorize trading symbols into
        their appropriate asset classes (Forex, Indices, Commodities, Stocks, Crypto).

        Args:
            symbols: List of trading symbols to classify

        Returns:
            Dictionary mapping each symbol to its asset class

        Example:
            >>> analyzer.classify_asset_symbols(['EURUSD', 'AAPL', 'XAUUSD'])
            {'EURUSD': 'Forex', 'AAPL': 'Stocks', 'XAUUSD': 'Commodities'}
        """
        logger.info(f"-> Engaging AI to classify {len(symbols)} asset symbols...")

        prompt = (
            "You are a financial data expert. Your task is to classify a list of trading symbols into their most specific asset class.\n\n"
            f"**SYMBOLS TO CLASSIFY:**\n{json.dumps(symbols, indent=2)}\n\n"
            "**INSTRUCTIONS:**\n"
            "1.  For each symbol, determine its asset class from the following options: 'Forex', 'Indices', 'Commodities', 'Stocks', 'Crypto'.\n"
            "2.  'XAUUSD' is 'Commodities', 'US30' is 'Indices', 'EURUSD' is 'Forex', etc.\n"
            "3.  Respond ONLY with a single, valid JSON object that maps each symbol string to its classification string.\n\n"
            "**EXAMPLE JSON RESPONSE:**\n"
            "```json\n"
            "{\n"
            '  "EURUSD": "Forex",\n'
            '  "XAUUSD": "Commodities",\n'
            '  "US30": "Indices",\n'
            '  "AAPL": "Stocks",\n'
            '  "BTCUSD": "Crypto"\n'
            "}\n"
            "```"
        )

        response_text = self._call_gemini(prompt)
        classified_assets = self._extract_json_from_response(response_text)

        if isinstance(classified_assets, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in classified_assets.items()):
            logger.info("  - AI successfully classified asset symbols.")
            for symbol in symbols:
                if symbol not in classified_assets:
                    classified_assets[symbol] = "Unknown"
                    logger.warning(f"  - AI did not classify '{symbol}'. Marked as 'Unknown'.")
            return classified_assets

        logger.error("  - AI failed to return a valid symbol classification dictionary. Using fallback detection.")
        return {}

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

    def get_contract_sizes_for_assets(self, symbols: List[str]) -> Dict[str, float]:
        """
        Determines contract sizes for given symbols using AI analysis.
        """
        if not self.api_key_valid:
            logger.warning("AI not available for contract size determination. Using defaults.")
            return {symbol: 1.0 for symbol in symbols}

        prompt = f"""
        You are a financial markets expert. For each of the following trading symbols, determine the standard contract size:

        Symbols: {symbols}

        For each symbol, provide the contract size (e.g., 1 for stocks, 100000 for forex, etc.).

        Respond in JSON format:
        {{
            "symbol1": contract_size,
            "symbol2": contract_size,
            ...
        }}
        """

        try:
            response = self._call_gemini(prompt)
            contract_sizes = self._extract_json_from_response(response)

            # Validate and provide defaults
            result = {}
            for symbol in symbols:
                if symbol in contract_sizes and isinstance(contract_sizes[symbol], (int, float)):
                    result[symbol] = float(contract_sizes[symbol])
                else:
                    result[symbol] = 1.0  # Default

            return result
        except Exception as e:
            logger.error(f"Error getting contract sizes from AI: {e}")
            return {symbol: 1.0 for symbol in symbols}

    def establish_strategic_directive(self, historical_results: List[Dict], current_state) -> str:
        """Establishes a strategic directive for the upcoming cycle."""
        logger.info("-> Establishing strategic directive for the upcoming cycle...")

        if not self.api_key_valid:
            return "Focus on consistent performance with moderate risk management."

        prompt = f"""
        You are a senior trading strategist. Based on the historical performance and current operating state, establish a strategic directive for the next trading cycle.

        Historical Results: {historical_results[-5:] if historical_results else "No history available"}
        Current Operating State: {current_state.value if hasattr(current_state, 'value') else str(current_state)}

        Provide a concise strategic directive (1-2 sentences) that will guide the AI's decision-making for the upcoming cycle.
        Focus on risk management, performance optimization, and market adaptation.

        Respond with just the directive text, no JSON formatting needed.
        """

        try:
            response = self._call_gemini(prompt)
            directive = response.strip().strip('"').strip("'")
            logger.info(f"Strategic directive established: {directive}")
            return directive
        except Exception as e:
            logger.error(f"Error establishing strategic directive: {e}")
            return "Focus on consistent performance with moderate risk management."

    def get_initial_run_configuration(self, script_version: str, ledger: Dict, memory: Dict, playbook: Dict,
                                    health_report: Dict, directives: List[Dict], data_summary: Dict,
                                    diagnosed_regime: str, regime_champions: Dict, correlation_summary_for_ai: str,
                                    master_macro_list: Dict, prime_directive_str: str, num_features: int,
                                    num_samples: int) -> Dict:
        """
        Generates initial run configuration using AI analysis of all available context.
        """
        if not self.api_key_valid:
            logger.warning("AI not available for initial configuration. Using defaults.")
            return {
                "strategy_name": "default_strategy",
                "selected_features": [],
                "reasoning": "AI not available - using default configuration"
            }

        # Prepare context summary
        context_summary = f"""
        Script Version: {script_version}
        Data Health: {health_report.get('status', 'Unknown')}
        Market Regime: {diagnosed_regime}
        Prime Directive: {prime_directive_str}
        Available Features: {num_features}
        Data Samples: {num_samples}
        Correlation Summary: {correlation_summary_for_ai}
        """

        prompt = f"""
        You are an expert trading system architect. Based on the comprehensive context below, recommend an optimal initial configuration.

        {context_summary}

        Available Strategies: {list(playbook.keys()) if playbook else "None"}

        Recommend:
        1. Best strategy for current conditions
        2. Key features to focus on (max 20)
        3. Risk management approach
        4. Reasoning for your choices

        Respond in JSON format:
        {{
            "strategy_name": "recommended_strategy",
            "selected_features": ["feature1", "feature2", ...],
            "risk_approach": "conservative/moderate/aggressive",
            "reasoning": "Your detailed reasoning..."
        }}
        """

        try:
            response = self._call_gemini(prompt)
            config = self._extract_json_from_response(response)

            # Validate response
            if not isinstance(config, dict):
                raise ValueError("Invalid response format")

            return {
                "strategy_name": config.get("strategy_name", "default_strategy"),
                "selected_features": config.get("selected_features", []),
                "risk_approach": config.get("risk_approach", "moderate"),
                "reasoning": config.get("reasoning", "AI recommendation")
            }

        except Exception as e:
            logger.error(f"Error getting initial configuration from AI: {e}")
            return {
                "strategy_name": "default_strategy",
                "selected_features": [],
                "reasoning": f"AI error: {e}"
            }

    def propose_capital_adjustment(self, total_equity: float, cycle_history: List[Dict], current_state, prime_directive: str) -> Dict:
        """
        Proposes capital allocation adjustments based on performance and state.
        """
        if not self.api_key_valid:
            return {"adjustment_factor": 1.0, "reasoning": "AI not available"}

        prompt = f"""
        You are a risk management expert. Based on the trading performance and current state, recommend capital allocation adjustments.

        Current Equity: ${total_equity:,.2f}
        Recent Performance: {cycle_history[-3:] if cycle_history else "No history"}
        Operating State: {current_state.value if hasattr(current_state, 'value') else str(current_state)}
        Prime Directive: {prime_directive}

        Recommend an adjustment factor (0.5 to 2.0) for position sizing:
        - 1.0 = no change
        - <1.0 = reduce position sizes
        - >1.0 = increase position sizes

        Respond in JSON format:
        {{
            "adjustment_factor": 1.0,
            "reasoning": "Your reasoning for the adjustment"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            result = self._extract_json_from_response(response)

            # Validate and constrain the adjustment factor
            factor = float(result.get("adjustment_factor", 1.0))
            factor = max(0.5, min(2.0, factor))  # Constrain between 0.5 and 2.0

            return {
                "adjustment_factor": factor,
                "reasoning": result.get("reasoning", "AI recommendation")
            }

        except Exception as e:
            logger.error(f"Error getting capital adjustment from AI: {e}")
            return {"adjustment_factor": 1.0, "reasoning": f"AI error: {e}"}

    def determine_optimal_label_quantiles(self, signal_pressure_summary: Dict, prime_directive: str) -> Dict:
        """
        Determines optimal quantiles for label creation based on signal pressure analysis.
        """
        if not self.api_key_valid:
            return {"quantiles": [0.33, 0.67], "reasoning": "AI not available"}

        prompt = f"""
        You are a quantitative analyst. Based on the signal pressure analysis, determine optimal quantiles for creating trading labels.

        Signal Pressure Summary: {signal_pressure_summary}
        Prime Directive: {prime_directive}

        Recommend quantiles that will create balanced, meaningful trading labels (Short/Hold/Long).
        Consider market volatility and signal strength distribution.

        Respond in JSON format:
        {{
            "quantiles": [lower_quantile, upper_quantile],
            "reasoning": "Your reasoning for these quantiles"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            result = self._extract_json_from_response(response)

            # Validate quantiles
            quantiles = result.get("quantiles", [0.33, 0.67])
            if len(quantiles) == 2 and 0 < quantiles[0] < quantiles[1] < 1:
                return result
            else:
                return {"quantiles": [0.33, 0.67], "reasoning": "Invalid quantiles from AI, using defaults"}

        except Exception as e:
            logger.error(f"Error determining optimal quantiles: {e}")
            return {"quantiles": [0.33, 0.67], "reasoning": f"AI error: {e}"}

    def determine_dynamic_f1_gate(self, optuna_summary: Dict, label_dist_summary: str, prime_directive: str, best_f1_from_validation: float) -> Dict:
        """
        Determines dynamic F1 score gate based on optimization results and label distribution.
        """
        if not self.api_key_valid:
            return {"f1_gate": 0.55, "reasoning": "AI not available"}

        prompt = f"""
        You are a machine learning expert. Based on the optimization results and label distribution, determine an appropriate F1 score gate.

        Optuna Summary: {optuna_summary}
        Label Distribution: {label_dist_summary}
        Prime Directive: {prime_directive}
        Best F1 from Validation: {best_f1_from_validation:.4f}

        Recommend an F1 score threshold that balances model quality with practical trading requirements.

        Respond in JSON format:
        {{
            "f1_gate": 0.XX,
            "reasoning": "Your reasoning for this threshold"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            result = self._extract_json_from_response(response)

            # Validate F1 gate
            f1_gate = result.get("f1_gate", 0.55)
            if 0.3 <= f1_gate <= 0.9:
                return result
            else:
                return {"f1_gate": 0.55, "reasoning": "Invalid F1 gate from AI, using default"}

        except Exception as e:
            logger.error(f"Error determining dynamic F1 gate: {e}")
            return {"f1_gate": 0.55, "reasoning": f"AI error: {e}"}

    def determine_dynamic_volatility_filter(self, df_train_labeled: pd.DataFrame) -> Dict:
        """
        Determines dynamic volatility filter based on training data characteristics.
        """
        if not self.api_key_valid:
            return {"volatility_threshold": 0.02, "reasoning": "AI not available"}

        # Calculate basic volatility statistics
        if 'ATR' in df_train_labeled.columns:
            atr_stats = {
                "mean": df_train_labeled['ATR'].mean(),
                "std": df_train_labeled['ATR'].std(),
                "median": df_train_labeled['ATR'].median(),
                "percentiles": {
                    "25th": df_train_labeled['ATR'].quantile(0.25),
                    "75th": df_train_labeled['ATR'].quantile(0.75),
                    "90th": df_train_labeled['ATR'].quantile(0.90)
                }
            }
        else:
            atr_stats = {"error": "ATR column not available"}

        prompt = f"""
        You are a risk management expert. Based on the training data volatility characteristics, determine an appropriate volatility filter.

        ATR Statistics: {atr_stats}
        Data Shape: {df_train_labeled.shape}

        Recommend a volatility threshold that filters out extreme market conditions while preserving sufficient training data.

        Respond in JSON format:
        {{
            "volatility_threshold": 0.XX,
            "reasoning": "Your reasoning for this threshold"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            result = self._extract_json_from_response(response)

            # Validate threshold
            threshold = result.get("volatility_threshold", 0.02)
            if 0.005 <= threshold <= 0.1:
                return result
            else:
                return {"volatility_threshold": 0.02, "reasoning": "Invalid threshold from AI, using default"}

        except Exception as e:
            logger.error(f"Error determining volatility filter: {e}")
            return {"volatility_threshold": 0.02, "reasoning": f"AI error: {e}"}

    def generate_scenario_analysis(self, failure_history: List[Dict], pre_analysis_summary: str, current_config: Dict, playbook: Dict, quarantine_list: List[str], available_features: List[str], failure_reason: str) -> Dict:
        """
        Generates scenario analysis for training failures and optimization issues.
        """
        if not self.api_key_valid:
            return {"scenarios": [], "reasoning": "AI not available"}

        prompt = f"""
        You are an AI doctor for trading systems. Analyze the training failure and generate recovery scenarios.

        Failure Reason: {failure_reason}
        Pre-Analysis Summary: {pre_analysis_summary}
        Current Config: {str(current_config)[:1000]}...
        Available Features: {len(available_features)} features
        Quarantined Strategies: {quarantine_list}

        Generate 3-5 specific scenarios to fix the training failure, ranked by likelihood of success.

        Respond in JSON format:
        {{
            "scenarios": [
                {{
                    "name": "Scenario name",
                    "changes": {{"param1": "value1", "param2": "value2"}},
                    "reasoning": "Why this might work",
                    "success_probability": 0.XX
                }}
            ],
            "analysis": "Overall analysis of the failure"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error generating scenario analysis: {e}")
            return {"scenarios": [], "reasoning": f"AI error: {e}"}

    def make_final_decision_from_scenarios(self, scenario_analysis: Dict, strategic_directive: str) -> Dict:
        """
        Makes final decision from scenario analysis for emergency interventions.
        """
        if not self.api_key_valid:
            return {}

        prompt = f"""
        You are an AI decision maker. Based on the scenario analysis, make a final decision for emergency intervention.

        Scenario Analysis: {scenario_analysis}
        Strategic Directive: {strategic_directive}

        Select the best scenario and provide specific parameter changes to implement.

        Respond in JSON format with the exact parameter changes:
        {{
            "selected_scenario": "scenario_name",
            "param1": "value1",
            "param2": "value2",
            "analysis_notes": "Reasoning for this decision"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error making final decision: {e}")
            return {}

    def select_best_tradeoff(self, best_trials: List, risk_profile: str, strategic_directive: str) -> Dict:
        """
        Selects the best tradeoff from Optuna trials based on risk profile and strategic directive.
        """
        if not self.api_key_valid:
            return {"selected_trial": 0, "reasoning": "AI not available"}

        # Convert trials to a summary format
        trial_summaries = []
        for i, trial in enumerate(best_trials[:5]):  # Limit to top 5
            if hasattr(trial, 'values') and trial.values:
                trial_summaries.append({
                    "trial_id": i,
                    "scores": trial.values,
                    "params": trial.params if hasattr(trial, 'params') else {}
                })

        prompt = f"""
        You are a portfolio manager. Select the best trial based on the risk profile and strategic directive.

        Trial Summaries: {trial_summaries}
        Risk Profile: {risk_profile}
        Strategic Directive: {strategic_directive}

        Select the trial that best balances performance with risk management.

        Respond in JSON format:
        {{
            "selected_trial": 0,
            "reasoning": "Why this trial is optimal for the given risk profile"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            result = self._extract_json_from_response(response)

            # Validate selection
            selected = result.get("selected_trial", 0)
            if 0 <= selected < len(best_trials):
                return result
            else:
                return {"selected_trial": 0, "reasoning": "Invalid selection from AI, using first trial"}

        except Exception as e:
            logger.error(f"Error selecting best tradeoff: {e}")
            return {"selected_trial": 0, "reasoning": f"AI error: {e}"}

    def propose_post_cycle_changes(self, cycle_performance: Dict, cycle_config: Dict, shap_summary: Optional[pd.DataFrame], strategic_directive: str) -> Dict:
        """
        Proposes post-cycle changes based on performance analysis.
        """
        if not self.api_key_valid:
            return {"changes": {}, "reasoning": "AI not available"}

        # Prepare SHAP summary if available
        shap_info = "No SHAP data available"
        if shap_summary is not None and not shap_summary.empty:
            top_features = shap_summary.head(10).to_dict('records')
            shap_info = f"Top 10 features by importance: {top_features}"

        prompt = f"""
        You are a trading system optimizer. Based on the cycle performance, propose specific changes for the next cycle.

        Cycle Performance: {cycle_performance}
        Strategic Directive: {strategic_directive}
        SHAP Analysis: {shap_info}

        Propose specific parameter changes to improve performance in the next cycle.

        Respond in JSON format:
        {{
            "changes": {{"param1": "value1", "param2": "value2"}},
            "reasoning": "Detailed reasoning for these changes",
            "expected_impact": "Expected impact on performance"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing post-cycle changes: {e}")
            return {"changes": {}, "reasoning": f"AI error: {e}"}

    def propose_holistic_cycle_update(self, cycle_performance: Dict, cycle_config: Dict, strategic_directive: str, diagnosed_regime: str, enhanced_shap_summary: Optional[pd.DataFrame] = None) -> Dict:
        """
        Proposes holistic cycle updates considering all available context.
        """
        if not self.api_key_valid:
            return {"changes": {}, "reasoning": "AI not available"}

        # Prepare enhanced SHAP summary if available
        shap_info = "No enhanced SHAP data available"
        if enhanced_shap_summary is not None and not enhanced_shap_summary.empty:
            feature_importance = enhanced_shap_summary.head(15).to_dict('records')
            shap_info = f"Enhanced feature importance: {feature_importance}"

        prompt = f"""
        You are a senior trading strategist. Conduct a holistic analysis and propose comprehensive updates for the next cycle.

        Cycle Performance: {cycle_performance}
        Current Config: {str(cycle_config)[:1000]}...
        Strategic Directive: {strategic_directive}
        Market Regime: {diagnosed_regime}
        Enhanced SHAP Analysis: {shap_info}

        Propose holistic changes that consider market regime, performance patterns, and feature importance.

        Respond in JSON format:
        {{
            "changes": {{"param1": "value1", "param2": "value2"}},
            "analysis_notes": "Comprehensive analysis of the situation",
            "regime_considerations": "How the market regime influenced decisions",
            "confidence_level": 0.XX
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing holistic cycle update: {e}")
            return {"changes": {}, "reasoning": f"AI error: {e}"}

    def propose_alpha_features(self, base_features: List[str], strategy_name: str, diagnosed_regime: str) -> Dict[str, Dict[str, str]]:
        """
        Engages the AI to invent new 'meta-features' based on existing base features.
        """
        if not self.api_key_valid:
            return {}

        prompt = f"""
        You are a quantitative researcher. Based on the existing features, invent new alpha-generating meta-features.

        Base Features: {base_features[:50]}  # Limit for prompt size
        Strategy: {strategy_name}
        Market Regime: {diagnosed_regime}

        Invent 5-10 novel meta-features that could provide trading alpha. Include mathematical formulas.

        Respond in JSON format:
        {{
            "feature_name_1": {{
                "formula": "mathematical formula using base features",
                "description": "what this feature captures",
                "rationale": "why it might provide alpha"
            }},
            "feature_name_2": {{...}}
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing alpha features: {e}")
            return {}

    def define_optuna_search_space(self, strategy_name: str, diagnosed_regime: str) -> Dict:
        """
        Asks the AI to propose an optimal Optuna search space for the given context.
        """
        if not self.api_key_valid:
            return {}

        prompt = f"""
        You are a hyperparameter optimization expert. Define an optimal Optuna search space for the given strategy and market regime.

        Strategy: {strategy_name}
        Market Regime: {diagnosed_regime}

        Define search spaces for key hyperparameters like learning rates, regularization, model complexity, etc.

        Respond in JSON format with Optuna-compatible parameter definitions:
        {{
            "learning_rate": {{"type": "float", "low": 0.001, "high": 0.1, "log": true}},
            "n_estimators": {{"type": "int", "low": 50, "high": 500}},
            "max_depth": {{"type": "int", "low": 3, "high": 15}},
            "reasoning": "Why these ranges are optimal for this context"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error defining Optuna search space: {e}")
            return {}

    def propose_strategic_intervention(self, historical_telemetry: List[Dict], playbook: Dict, intervention_history: Dict) -> Dict:
        """
        Proposes strategic interventions based on historical performance and telemetry.
        """
        if not self.api_key_valid:
            return {}

        # Summarize recent performance
        recent_performance = []
        for entry in historical_telemetry[-5:]:
            if 'performance' in entry:
                recent_performance.append({
                    'cycle': entry.get('cycle_num', 'unknown'),
                    'metrics': entry['performance']
                })

        prompt = f"""
        You are a strategic advisor for trading systems. Based on historical performance and intervention history, propose strategic interventions.

        Recent Performance: {recent_performance}
        Available Strategies: {list(playbook.keys())}
        Previous Interventions: {str(intervention_history)[:500]}...

        Propose strategic interventions to improve overall framework performance.

        Respond in JSON format:
        {{
            "intervention_type": "STRATEGIC_ADJUSTMENT",
            "proposed_changes": {{"param1": "value1", "param2": "value2"}},
            "reasoning": "Strategic reasoning for intervention",
            "expected_outcome": "Expected improvement",
            "risk_assessment": "Potential risks and mitigation"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing strategic intervention: {e}")
            return {}

    def propose_playbook_amendment(self, quarantined_strategy_name: str, framework_history: Dict, playbook: Dict) -> Dict:
        """
        Proposes amendments to the strategy playbook when strategies are quarantined.
        """
        if not self.api_key_valid:
            return {}

        prompt = f"""
        You are a strategy architect. A strategy has been quarantined due to poor performance. Propose how to amend the playbook.

        Quarantined Strategy: {quarantined_strategy_name}
        Current Playbook: {list(playbook.keys())}
        Framework History: {str(framework_history)[:500]}...

        Decide whether to rework the strategy or retire it, and provide specific amendments.

        Respond in JSON format:
        {{
            "action": "rework" or "retire",
            "new_config": {{"param1": "value1"}} if reworking,
            "reasoning": "Why this action is recommended",
            "alternative_strategies": ["strategy1", "strategy2"] if retiring
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing playbook amendment: {e}")
            return {}

    def propose_regime_based_strategy_switch(self, regime_data: Dict, playbook: Dict, current_strategy_name: str, quarantine_list: List[str]) -> Dict:
        """
        Proposes strategy switches based on detected market regime changes.
        """
        if not self.api_key_valid:
            return {}

        available_strategies = [name for name in playbook.keys() if name not in quarantine_list and name != current_strategy_name]

        prompt = f"""
        You are a regime-aware strategy selector. Based on the detected market regime, recommend whether to switch strategies.

        Current Regime Data: {regime_data}
        Current Strategy: {current_strategy_name}
        Available Strategies: {available_strategies}
        Quarantined Strategies: {quarantine_list}

        Decide if a strategy switch is warranted and recommend the best alternative.

        Respond in JSON format:
        {{
            "action": "SWITCH" or "MAINTAIN",
            "new_strategy_name": "strategy_name" if switching,
            "reasoning": "Why this decision is optimal for the current regime",
            "confidence": 0.XX
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing regime-based strategy switch: {e}")
            return {}

    def propose_new_playbook_strategy(self, failed_strategy_name: str, playbook: Dict, framework_history: Dict) -> Dict:
        """
        Proposes new strategies to add to the playbook when existing ones fail.
        """
        if not self.api_key_valid:
            return {}

        prompt = f"""
        You are a strategy inventor. A strategy has failed and been retired. Invent a new strategy to add to the playbook.

        Failed Strategy: {failed_strategy_name}
        Existing Strategies: {list(playbook.keys())}
        Framework History: {str(framework_history)[:500]}...

        Invent a novel trading strategy with specific parameters and configuration.

        Respond in JSON format:
        {{
            "new_strategy_name": {{
                "description": "Strategy description",
                "parameters": {{"param1": "value1", "param2": "value2"}},
                "risk_profile": "conservative/moderate/aggressive",
                "market_conditions": "Best market conditions for this strategy",
                "innovation": "What makes this strategy unique"
            }}
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing new playbook strategy: {e}")
            return {}

    def propose_feature_quarantine(self, shap_history: List[Dict], current_features: List[str]) -> Dict:
        """
        Analyzes SHAP history to identify and quarantine features with decaying importance.
        """
        if not self.api_key_valid:
            return {"quarantine_features": [], "reasoning": "AI not available"}

        prompt = f"""
        You are a feature engineering expert. Based on SHAP importance history, identify features that should be quarantined.

        SHAP History: {shap_history[-10:]}  # Last 10 entries
        Current Features: {len(current_features)} features

        Identify features with consistently declining importance that should be quarantined.

        Respond in JSON format:
        {{
            "quarantine_features": ["feature1", "feature2"],
            "reasoning": "Why these features should be quarantined",
            "impact_assessment": "Expected impact of quarantining these features"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing feature quarantine: {e}")
            return {"quarantine_features": [], "reasoning": f"AI error: {e}"}

    def propose_gp_failure_fallback(self, playbook: Dict, quarantine_list: List[str]) -> Dict:
        """
        Proposes fallback strategies when genetic programming fails.
        """
        if not self.api_key_valid:
            return {}

        available_strategies = [name for name in playbook.keys() if name not in quarantine_list]

        prompt = f"""
        You are a strategy recovery expert. Genetic programming has failed to find profitable rules. Recommend a fallback strategy.

        Available Strategies: {available_strategies}
        Quarantined Strategies: {quarantine_list}

        Recommend the most robust fallback strategy for when evolutionary approaches fail.

        Respond in JSON format:
        {{
            "fallback_strategy_name": "strategy_name",
            "reasoning": "Why this strategy is the best fallback",
            "risk_mitigation": "How this strategy mitigates the GP failure risk"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing GP failure fallback: {e}")
            return {}

    def define_gene_pool(self, strategy_goal: str, available_features: List[str]) -> Dict:
        """
        Defines the gene pool for genetic programming evolution.
        """
        if not self.api_key_valid:
            return {}

        prompt = f"""
        You are a genetic programming expert. Define a gene pool for evolving trading rules.

        Strategy Goal: {strategy_goal}
        Available Features: {available_features[:30]}  # Limit for prompt size

        Define the genetic components: features, operators, and constants for rule evolution.

        Respond in JSON format:
        {{
            "continuous_features": ["feature1", "feature2"],
            "state_features": ["feature3", "feature4"],
            "comparison_operators": [">", "<", ">=", "<="],
            "state_operators": ["==", "!="],
            "logical_operators": ["AND", "OR"],
            "constants": [0, 25, 50, 75, 100],
            "reasoning": "Why this gene pool is optimal for the strategy goal"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error defining gene pool: {e}")
            return {}

    def review_horizon_performance(self, horizon_performance_history: Dict) -> Dict:
        """
        Reviews performance across different prediction horizons and provides directives.
        """
        if not self.api_key_valid:
            return {}

        prompt = f"""
        You are a multi-horizon trading analyst. Review the performance across different prediction horizons and provide management directives.

        Horizon Performance History: {horizon_performance_history}

        Analyze which horizons are performing well and provide specific directives for model management.

        Respond in JSON format:
        {{
            "horizon_analysis": {{"h30": "analysis", "h60": "analysis"}},
            "directives": {{"h30": "directive", "h60": "directive"}},
            "recommendations": "Overall recommendations for horizon management",
            "focus_horizon": "Which horizon to prioritize"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error reviewing horizon performance: {e}")
            return {}

    def propose_horizon_specific_search_spaces(self, horizons: List[int], strategy_name: str, diagnosed_regime: str) -> Dict[str, Dict]:
        """
        Asks the AI to propose an optimal Optuna search space for EACH horizon.
        """
        if not self.api_key_valid:
            return {}

        prompt = f"""
        You are a multi-horizon optimization expert. Define optimal Optuna search spaces for each prediction horizon.

        Horizons: {horizons}
        Strategy: {strategy_name}
        Market Regime: {diagnosed_regime}

        Define horizon-specific search spaces that account for the different prediction timeframes.

        Respond in JSON format:
        {{
            "h30": {{"learning_rate": {{"type": "float", "low": 0.001, "high": 0.1}}}},
            "h60": {{"learning_rate": {{"type": "float", "low": 0.001, "high": 0.1}}}},
            "reasoning": "Why different horizons need different search spaces"
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing horizon-specific search spaces: {e}")
            return {}

    def propose_gp_gene_pool_fix(self, failed_gene_pool: Dict, evolution_log: List[Dict], all_available_features: List[str]) -> Dict:
        """
        Proposes fixes to the gene pool when genetic programming evolution fails.
        """
        if not self.api_key_valid:
            return {}

        prompt = f"""
        You are a genetic programming debugger. The gene pool failed to evolve profitable rules. Propose fixes.

        Failed Gene Pool: {failed_gene_pool}
        Evolution Log: {evolution_log[-5:]}  # Last 5 entries
        Available Features: {all_available_features[:20]}  # Limit for prompt size

        Propose a revised gene pool that addresses the evolution failure.

        Respond in JSON format:
        {{
            "revised_gene_pool": {{
                "continuous_features": ["feature1", "feature2"],
                "state_features": ["feature3", "feature4"],
                "comparison_operators": [">", "<"],
                "constants": [0, 25, 50, 75, 100]
            }},
            "fixes_applied": "What was changed and why",
            "success_probability": 0.XX
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error proposing GP gene pool fix: {e}")
            return {}

    def discover_behavioral_patterns(self, base_features: List[str], diagnosed_regime: str) -> Dict[str, Dict[str, str]]:
        """
        Engages the AI to discover novel non-technical or behavioral patterns.
        """
        if not self.api_key_valid:
            return {}

        prompt = f"""
        You are a behavioral finance expert. Based on the available features, discover novel behavioral trading patterns.

        Base Features: {base_features[:30]}  # Limit for prompt size
        Market Regime: {diagnosed_regime}

        Discover behavioral patterns that go beyond traditional technical analysis.

        Respond in JSON format:
        {{
            "pattern_name_1": {{
                "description": "What behavioral pattern this captures",
                "formula": "How to calculate this pattern",
                "behavioral_basis": "The psychological/behavioral foundation"
            }},
            "pattern_name_2": {{...}}
        }}
        """

        try:
            response = self._call_gemini(prompt)
            return self._extract_json_from_response(response)
        except Exception as e:
            logger.error(f"Error discovering behavioral patterns: {e}")
            return {}


class InterventionManager:
    """
    Enhanced AI intervention tracking system for monitoring and evaluating AI suggestions.

    Tracks AI suggestions, links them to cycle performance, and provides a historical
    record for a robust feedback loop. This enables the framework to learn from past
    AI interventions and improve future suggestions.

    Features:
    - Unique intervention ID generation and tracking
    - Performance impact evaluation through before/after comparison
    - Historical feedback generation for AI prompts
    - Persistent storage of intervention outcomes
    - Comprehensive intervention analytics
    """

    def __init__(self, ledger_path: str):
        """
        Initialize intervention manager with persistent storage.

        Args:
            ledger_path: Path to JSON file for storing intervention history
        """
        self.ledger_path = ledger_path
        self.intervention_ledger: Dict[str, Any] = self._load_ledger()
        logger.info(f"Intervention Manager initialized. Loaded {len(self.intervention_ledger)} past interventions.")

    def _load_ledger(self) -> Dict[str, Any]:
        """
        Load intervention history from JSON file.

        Returns:
            Dictionary containing intervention history
        """
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, 'r') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError):
                logger.error(f"Could not load intervention ledger from {self.ledger_path}. Starting fresh.")
        return {}

    def _save_ledger(self):
        """Save current intervention ledger state to file."""
        try:
            with open(self.ledger_path, 'w') as f:
                json.dump(self.intervention_ledger, f, indent=4)
        except IOError:
            logger.error(f"Could not save intervention ledger to {self.ledger_path}.")

    def log_intervention(self, cycle_num: int, suggestion_type: str, scope: str,
                        regime: str, details: Dict, notes: str) -> str:
        """
        Log a new AI intervention with unique tracking ID.

        Args:
            cycle_num: Cycle number where intervention was applied
            suggestion_type: Type of AI suggestion (e.g., 'parameter_adjustment', 'strategy_change')
            scope: Scope of intervention (e.g., 'strategy_EmaCrossover', 'feature_engineering')
            regime: Market regime at time of suggestion
            details: Detailed information about the intervention
            notes: AI rationale for the suggestion

        Returns:
            Unique intervention ID for tracking
        """
        import uuid

        intervention_id = f"INT-{cycle_num}-{uuid.uuid4().hex[:8]}"
        self.intervention_ledger[intervention_id] = {
            "intervention_id": intervention_id,
            "applied_to_cycle": cycle_num,
            "type": suggestion_type,
            "scope": scope,  # e.g., "strategy_EmaCrossover", "feature_engineering"
            "regime_at_suggestion": regime,
            "suggestion_details": details,
            "ai_rationale": notes,
            "status": "applied",  # Mark as applied immediately
            "performance_impact": {}
        }
        self._save_ledger()
        logger.info(f"Logged new AI intervention {intervention_id} for cycle {cycle_num}.")
        return intervention_id

    def evaluate_intervention_outcome(self, intervention_id: str, historical_telemetry: List[Dict]):
        """
        Evaluate performance impact of AI intervention by comparing before/after metrics.

        Args:
            intervention_id: Unique ID of intervention to evaluate
            historical_telemetry: List of historical cycle telemetry data
        """
        if intervention_id not in self.intervention_ledger:
            logger.warning(f"Cannot evaluate outcome: Intervention ID {intervention_id} not found.")
            return

        intervention = self.intervention_ledger[intervention_id]
        cycle_num = intervention["applied_to_cycle"]

        # Find the cycle where the intervention was applied and the one before it
        current_cycle_telemetry = next((c for c in historical_telemetry if c.get('cycle') == cycle_num), None)
        previous_cycle_telemetry = next((c for c in historical_telemetry if c.get('cycle') == cycle_num - 1), None)

        if not current_cycle_telemetry or not previous_cycle_telemetry:
            logger.warning(f"Cannot evaluate intervention {intervention_id}: Missing telemetry for comparison.")
            return

        current_metrics = current_cycle_telemetry.get("performance", {})
        previous_metrics = previous_cycle_telemetry.get("performance", {})

        # Calculate performance delta
        delta_sharpe = current_metrics.get("sharpe_ratio", 0) - previous_metrics.get("sharpe_ratio", 0)
        delta_mar = current_metrics.get("mar_ratio", 0) - previous_metrics.get("mar_ratio", 0)
        # Lower is better for drawdown
        delta_max_dd = current_metrics.get("max_drawdown_pct", 100) - previous_metrics.get("max_drawdown_pct", 100)

        impact = {
            "before_sharpe": previous_metrics.get("sharpe_ratio", 0),
            "after_sharpe": current_metrics.get("sharpe_ratio", 0),
            "delta_sharpe": delta_sharpe,
            "before_mar": previous_metrics.get("mar_ratio", 0),
            "after_mar": current_metrics.get("mar_ratio", 0),
            "delta_mar": delta_mar,
            "before_max_dd_pct": previous_metrics.get("max_drawdown_pct", 100),
            "after_max_dd_pct": current_metrics.get("max_drawdown_pct", 100),
            "delta_max_dd_pct": delta_max_dd
        }

        self.intervention_ledger[intervention_id]['performance_impact'] = impact
        self.intervention_ledger[intervention_id]['status'] = 'evaluated'
        self._save_ledger()
        logger.info(f"Evaluated AI intervention {intervention_id}. Delta Sharpe: {delta_sharpe:+.3f}, Delta MAR: {delta_mar:+.3f}")

    def get_feedback_for_ai_prompt(self, suggestion_type: str, scope: str, num_to_include: int = 5) -> str:
        """
        Generate text summary of past intervention outcomes for AI prompt feedback.

        Args:
            suggestion_type: Type of suggestion to get feedback for
            scope: Scope of intervention to filter by
            num_to_include: Number of recent interventions to include

        Returns:
            Formatted feedback summary for AI prompts
        """
        relevant_history = [
            v for v in self.intervention_ledger.values()
            if v['status'] == 'evaluated' and v['type'] == suggestion_type and v['scope'] == scope
        ]

        if not relevant_history:
            return "No historical performance data for this type of suggestion."

        # Sort by cycle number to get the most recent
        sorted_history = sorted(relevant_history, key=lambda x: x['applied_to_cycle'], reverse=True)

        feedback_summary = "\n## Historical AI Suggestion Performance (Most Recent):\n"
        for item in sorted_history[:num_to_include]:
            impact = item.get('performance_impact', {})
            details = item.get('suggestion_details', {})
            delta_mar = impact.get('delta_mar', 0)

            outcome = "IMPROVED" if delta_mar > 0.1 else "WORSENED" if delta_mar < -0.1 else "NEUTRAL"

            feedback_summary += f"- Suggestion: {details}\n"
            feedback_summary += f"  - Outcome: **{outcome}** (MAR Ratio change: {delta_mar:+.3f})\n"

        return feedback_summary

    def get_intervention_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about intervention performance.

        Returns:
            Dictionary containing intervention statistics
        """
        try:
            evaluated_interventions = [v for v in self.intervention_ledger.values() if v['status'] == 'evaluated']

            if not evaluated_interventions:
                return {'total_interventions': 0, 'evaluated_interventions': 0}

            # Calculate success metrics
            successful_interventions = [
                v for v in evaluated_interventions
                if v.get('performance_impact', {}).get('delta_mar', 0) > 0.1
            ]

            failed_interventions = [
                v for v in evaluated_interventions
                if v.get('performance_impact', {}).get('delta_mar', 0) < -0.1
            ]

            # Calculate average impacts
            avg_sharpe_delta = np.mean([
                v.get('performance_impact', {}).get('delta_sharpe', 0)
                for v in evaluated_interventions
            ])

            avg_mar_delta = np.mean([
                v.get('performance_impact', {}).get('delta_mar', 0)
                for v in evaluated_interventions
            ])

            return {
                'total_interventions': len(self.intervention_ledger),
                'evaluated_interventions': len(evaluated_interventions),
                'successful_interventions': len(successful_interventions),
                'failed_interventions': len(failed_interventions),
                'success_rate': len(successful_interventions) / len(evaluated_interventions) if evaluated_interventions else 0,
                'avg_sharpe_delta': avg_sharpe_delta,
                'avg_mar_delta': avg_mar_delta,
                'intervention_types': list(set(v['type'] for v in evaluated_interventions)),
                'intervention_scopes': list(set(v['scope'] for v in evaluated_interventions))
            }

        except Exception as e:
            logger.error(f"Error calculating intervention statistics: {e}")
            return {'error': str(e)}

    def get_recent_interventions(self, num_recent: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent interventions for analysis.

        Args:
            num_recent: Number of recent interventions to return

        Returns:
            List of recent intervention records
        """
        try:
            all_interventions = list(self.intervention_ledger.values())
            sorted_interventions = sorted(all_interventions, key=lambda x: x['applied_to_cycle'], reverse=True)
            return sorted_interventions[:num_recent]

        except Exception as e:
            logger.error(f"Error getting recent interventions: {e}")
            return []

    def clear_old_interventions(self, keep_recent: int = 100):
        """
        Clear old interventions to prevent ledger from growing too large.

        Args:
            keep_recent: Number of recent interventions to keep
        """
        try:
            if len(self.intervention_ledger) <= keep_recent:
                return

            # Sort by cycle number and keep only the most recent
            sorted_items = sorted(self.intervention_ledger.items(),
                                key=lambda x: x[1]['applied_to_cycle'], reverse=True)

            # Keep only the most recent interventions
            self.intervention_ledger = dict(sorted_items[:keep_recent])
            self._save_ledger()

            logger.info(f"Cleared old interventions, keeping {len(self.intervention_ledger)} most recent.")

        except Exception as e:
            logger.error(f"Error clearing old interventions: {e}")


def optimize_strategy_with_ai(analyzer, strategy_name: str, current_performance: Dict,
                             historical_data: Dict, market_regime: str) -> Dict[str, Any]:
    """
    Use AI to optimize strategy parameters based on current performance and market conditions.

    Args:
        analyzer: GeminiAnalyzer instance
        strategy_name: Name of the strategy to optimize
        current_performance: Current strategy performance metrics
        historical_data: Historical performance data
        market_regime: Current market regime

    Returns:
        Dictionary containing AI optimization suggestions
    """
    try:
        if not analyzer.api_key_valid:
            logger.warning("AI analyzer not available for strategy optimization")
            return {}

        logger.info(f"-> Optimizing strategy '{strategy_name}' with AI for regime '{market_regime}'...")

        # Prepare context for AI
        performance_summary = {
            'mar_ratio': current_performance.get('mar_ratio', 0),
            'sharpe_ratio': current_performance.get('sharpe_ratio', 0),
            'max_drawdown_pct': current_performance.get('max_drawdown_pct', 0),
            'win_rate': current_performance.get('win_rate', 0),
            'profit_factor': current_performance.get('profit_factor', 0)
        }

        prompt = f"""
        You are an expert trading strategy optimizer. Analyze the current strategy performance and suggest optimizations.

        **STRATEGY:** {strategy_name}
        **CURRENT MARKET REGIME:** {market_regime}

        **CURRENT PERFORMANCE:**
        - MAR Ratio: {performance_summary['mar_ratio']:.3f}
        - Sharpe Ratio: {performance_summary['sharpe_ratio']:.3f}
        - Max Drawdown: {performance_summary['max_drawdown_pct']:.2f}%
        - Win Rate: {performance_summary['win_rate']:.2%}
        - Profit Factor: {performance_summary['profit_factor']:.3f}

        **YOUR TASK:**
        1. Identify performance weaknesses
        2. Suggest specific parameter adjustments
        3. Recommend feature modifications
        4. Provide regime-specific optimizations

        **OUTPUT FORMAT:** JSON object with optimization suggestions:
        ```json
        {{
            "parameter_adjustments": {{
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04,
                "position_size_pct": 0.1
            }},
            "feature_recommendations": ["RSI", "MACD", "ATR"],
            "regime_specific_settings": {{
                "confidence_threshold": 0.7,
                "max_positions": 3
            }},
            "optimization_rationale": "Explanation of suggested changes"
        }}
        ```
        """

        response = analyzer._call_gemini(prompt)
        optimization_suggestions = analyzer._extract_json_from_response(response)

        logger.info(f"AI optimization suggestions generated for {strategy_name}")
        return optimization_suggestions

    except Exception as e:
        logger.error(f"Error in AI strategy optimization: {e}")
        return {}


def generate_ai_insights(analyzer, performance_data: Dict, market_data: pd.DataFrame,
                        shap_summary: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Generate comprehensive AI insights from performance data and market analysis.

    Args:
        analyzer: GeminiAnalyzer instance
        performance_data: Performance metrics and trade data
        market_data: Market data DataFrame
        shap_summary: Optional SHAP feature importance summary

    Returns:
        Dictionary containing AI-generated insights
    """
    try:
        if not analyzer.api_key_valid:
            logger.warning("AI analyzer not available for insight generation")
            return {}

        logger.info("-> Generating comprehensive AI insights...")

        # Prepare market context
        recent_data = market_data.tail(50) if not market_data.empty else pd.DataFrame()
        market_summary = {}

        if not recent_data.empty:
            market_summary = {
                'volatility': recent_data['Close'].pct_change().std() * np.sqrt(252) if 'Close' in recent_data.columns else 0,
                'trend_strength': recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1 if 'Close' in recent_data.columns else 0,
                'volume_trend': recent_data['Volume'].mean() if 'Volume' in recent_data.columns else 0
            }

        # Prepare SHAP insights
        shap_insights = ""
        if shap_summary is not None and not shap_summary.empty:
            top_features = shap_summary.head(5)
            shap_insights = f"Top 5 important features: {', '.join(top_features.index.tolist())}"

        prompt = f"""
        You are an expert quantitative analyst. Provide comprehensive insights based on the trading performance and market data.

        **PERFORMANCE METRICS:**
        - Total Return: {performance_data.get('net_profit_pct', 0):.2%}
        - Sharpe Ratio: {performance_data.get('sharpe_ratio', 0):.3f}
        - Max Drawdown: {performance_data.get('max_drawdown_pct', 0):.2f}%
        - Win Rate: {performance_data.get('win_rate', 0):.2%}
        - Total Trades: {performance_data.get('total_trades', 0)}

        **MARKET CONTEXT:**
        - Recent Volatility: {market_summary.get('volatility', 0):.2%}
        - Trend Strength: {market_summary.get('trend_strength', 0):.2%}

        **FEATURE IMPORTANCE:**
        {shap_insights}

        **YOUR TASK:**
        Provide actionable insights covering:
        1. Performance strengths and weaknesses
        2. Market regime analysis
        3. Risk management assessment
        4. Feature effectiveness evaluation
        5. Strategic recommendations

        **OUTPUT FORMAT:** JSON object with structured insights:
        ```json
        {{
            "performance_assessment": {{
                "strengths": ["List of performance strengths"],
                "weaknesses": ["List of areas for improvement"],
                "overall_rating": "Excellent/Good/Fair/Poor"
            }},
            "market_analysis": {{
                "regime_diagnosis": "Current market regime assessment",
                "volatility_impact": "How volatility affects performance",
                "trend_alignment": "Strategy alignment with market trends"
            }},
            "risk_evaluation": {{
                "drawdown_analysis": "Assessment of drawdown patterns",
                "position_sizing": "Position sizing effectiveness",
                "risk_adjusted_returns": "Risk-adjusted performance evaluation"
            }},
            "feature_insights": {{
                "most_effective": ["Most effective features"],
                "underperforming": ["Features that may need adjustment"],
                "recommendations": ["Feature engineering suggestions"]
            }},
            "strategic_recommendations": {{
                "immediate_actions": ["Short-term improvements"],
                "long_term_strategy": ["Long-term strategic direction"],
                "regime_adaptations": ["Regime-specific adjustments"]
            }}
        }}
        ```
        """

        response = analyzer._call_gemini(prompt)
        insights = analyzer._extract_json_from_response(response)

        logger.info("Comprehensive AI insights generated successfully")
        return insights

    except Exception as e:
        logger.error(f"Error generating AI insights: {e}")
        return {}


def calculate_ai_confidence(model_predictions: np.ndarray, ensemble_weights: Dict[str, float] = None,
                          historical_accuracy: Dict[str, float] = None) -> float:
    """
    Calculate AI confidence score based on model predictions and historical performance.

    Args:
        model_predictions: Array of model prediction probabilities
        ensemble_weights: Optional weights for ensemble models
        historical_accuracy: Optional historical accuracy scores for models

    Returns:
        Confidence score between 0 and 1
    """
    try:
        if len(model_predictions) == 0:
            return 0.0

        # Base confidence from prediction probabilities
        max_proba = np.max(model_predictions)
        prediction_confidence = max_proba

        # Adjust for ensemble agreement if multiple predictions
        if len(model_predictions) > 1:
            # Calculate agreement between models
            prediction_variance = np.var(model_predictions)
            agreement_factor = 1.0 - min(prediction_variance * 4, 0.5)  # Cap variance penalty
            prediction_confidence *= agreement_factor

        # Incorporate ensemble weights if provided
        if ensemble_weights:
            weighted_confidence = 0.0
            total_weight = 0.0

            for i, (model_name, weight) in enumerate(ensemble_weights.items()):
                if i < len(model_predictions):
                    weighted_confidence += model_predictions[i] * weight
                    total_weight += weight

            if total_weight > 0:
                prediction_confidence = weighted_confidence / total_weight

        # Adjust for historical accuracy if provided
        if historical_accuracy:
            avg_accuracy = np.mean(list(historical_accuracy.values()))
            accuracy_factor = min(avg_accuracy * 1.2, 1.0)  # Boost for good historical performance
            prediction_confidence *= accuracy_factor

        # Ensure confidence is within valid range
        confidence = max(0.0, min(1.0, prediction_confidence))

        logger.debug(f"Calculated AI confidence: {confidence:.3f}")
        return confidence

    except Exception as e:
        logger.error(f"Error calculating AI confidence: {e}")
        return 0.5  # Default moderate confidence


def update_confidence_gate(config, diagnosed_regime: str, recent_performance: Dict[str, float] = None,
                          adaptive_factor: float = 0.1) -> float:
    """
    Update confidence gate based on market regime and recent performance.

    Args:
        config: Configuration object with regime confidence gates
        diagnosed_regime: Current market regime
        recent_performance: Optional recent performance metrics
        adaptive_factor: Factor for adaptive adjustments (0.0 to 1.0)

    Returns:
        Updated confidence gate value
    """
    try:
        # Get base confidence gate for regime
        regime_gates = getattr(config, 'REGIME_CONFIDENCE_GATES', {})
        static_gate = getattr(config, 'STATIC_CONFIDENCE_GATE', 0.70)

        # Normalize regime name to match keys
        normalized_regime = diagnosed_regime.strip().split('_')[-1].title()
        base_gate = regime_gates.get(normalized_regime, regime_gates.get('Default', static_gate))

        # Apply adaptive adjustments based on recent performance
        adjusted_gate = base_gate

        if recent_performance and adaptive_factor > 0:
            # Get performance metrics
            win_rate = recent_performance.get('win_rate', 0.5)
            sharpe_ratio = recent_performance.get('sharpe_ratio', 0.0)
            mar_ratio = recent_performance.get('mar_ratio', 0.0)

            # Calculate performance score (0 to 1)
            performance_score = (
                win_rate * 0.4 +  # Win rate contribution
                min(max(sharpe_ratio / 2.0, 0), 1) * 0.3 +  # Sharpe ratio contribution
                min(max(mar_ratio / 3.0, 0), 1) * 0.3  # MAR ratio contribution
            )

            # Adjust gate based on performance
            if performance_score > 0.7:
                # Good performance - can lower gate slightly for more trades
                adjustment = -adaptive_factor * 0.1
            elif performance_score < 0.3:
                # Poor performance - raise gate for more selectivity
                adjustment = adaptive_factor * 0.15
            else:
                # Moderate performance - minimal adjustment
                adjustment = 0.0

            adjusted_gate = base_gate + adjustment

        # Ensure gate is within reasonable bounds
        final_gate = max(0.5, min(0.95, adjusted_gate))

        logger.info(f"Updated confidence gate for regime '{diagnosed_regime}': {base_gate:.3f} -> {final_gate:.3f}")
        return final_gate

    except Exception as e:
        logger.error(f"Error updating confidence gate: {e}")
        return 0.70  # Default confidence gate


def get_regime_adaptive_confidence_gate(config, diagnosed_regime: str) -> float:
    """
    Retrieve the confidence gate value based on the diagnosed regime.

    Args:
        config: Configuration object with regime confidence gates
        diagnosed_regime: Current market regime

    Returns:
        Confidence gate value for the regime
    """
    try:
        regime_gates = getattr(config, 'REGIME_CONFIDENCE_GATES', {})
        static_gate = getattr(config, 'STATIC_CONFIDENCE_GATE', 0.70)

        # Normalize regime name to match keys (e.g., "Trending", "Ranging")
        normalized_regime = diagnosed_regime.strip().split('_')[-1].title()
        gate = regime_gates.get(normalized_regime, regime_gates.get('Default', static_gate))

        logger.debug(f"Using regime-adaptive confidence gate: {gate} for regime '{diagnosed_regime}'")
        return gate

    except Exception as e:
        logger.error(f"Error getting regime confidence gate: {e}")
        return 0.70  # Default confidence gate


def validate_ai_confidence_thresholds(config, performance_history: List[Dict] = None) -> Dict[str, float]:
    """
    Validate and suggest optimal confidence thresholds based on historical performance.

    Args:
        config: Configuration object
        performance_history: Optional historical performance data

    Returns:
        Dictionary of recommended confidence thresholds by regime
    """
    try:
        recommendations = {}

        # Get current regime gates
        regime_gates = getattr(config, 'REGIME_CONFIDENCE_GATES', {})

        # Analyze historical performance if available
        if performance_history:
            regime_performance = {}

            for run in performance_history:
                regime = run.get('diagnosed_regime', 'Unknown')
                metrics = run.get('final_metrics', {})

                if regime not in regime_performance:
                    regime_performance[regime] = []

                regime_performance[regime].append({
                    'mar_ratio': metrics.get('mar_ratio', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'total_trades': metrics.get('total_trades', 0)
                })

            # Calculate optimal thresholds for each regime
            for regime, runs in regime_performance.items():
                if len(runs) >= 3:  # Need sufficient data
                    avg_mar = np.mean([r['mar_ratio'] for r in runs])
                    avg_trades = np.mean([r['total_trades'] for r in runs])

                    # Suggest threshold based on performance
                    if avg_mar > 1.0 and avg_trades > 10:
                        # Good performance - can use lower threshold
                        recommended_threshold = 0.65
                    elif avg_mar > 0.5:
                        # Moderate performance - standard threshold
                        recommended_threshold = 0.70
                    else:
                        # Poor performance - higher threshold for selectivity
                        recommended_threshold = 0.75

                    recommendations[regime] = recommended_threshold

        # Fill in defaults for missing regimes
        default_regimes = ['Trending', 'Ranging', 'Highvolatility', 'Default']
        for regime in default_regimes:
            if regime not in recommendations:
                recommendations[regime] = regime_gates.get(regime, 0.70)

        logger.info(f"Confidence threshold recommendations: {recommendations}")
        return recommendations

    except Exception as e:
        logger.error(f"Error validating confidence thresholds: {e}")
        return {'Default': 0.70}


def switch_ai_model(analyzer, preferred_model: str = None, fallback_models: List[str] = None) -> bool:
    """
    Switch AI model to a preferred model or update the priority list.

    Args:
        analyzer: GeminiAnalyzer instance
        preferred_model: Preferred model to use
        fallback_models: List of fallback models

    Returns:
        True if model switch was successful, False otherwise
    """
    try:
        if not hasattr(analyzer, 'model_priority_list'):
            logger.error("Analyzer does not have model_priority_list attribute")
            return False

        # Default model priority list
        default_models = [
            "gemini-2.5-flash-lite-preview-06-17",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite"
        ]

        new_priority_list = []

        # Add preferred model first if specified
        if preferred_model:
            new_priority_list.append(preferred_model)
            logger.info(f"Setting preferred AI model: {preferred_model}")

        # Add fallback models
        if fallback_models:
            for model in fallback_models:
                if model not in new_priority_list:
                    new_priority_list.append(model)

        # Add default models that aren't already in the list
        for model in default_models:
            if model not in new_priority_list:
                new_priority_list.append(model)

        # Update the analyzer's model priority list
        analyzer.model_priority_list = new_priority_list
        logger.info(f"Updated AI model priority list: {new_priority_list}")

        return True

    except Exception as e:
        logger.error(f"Error switching AI model: {e}")
        return False


def validate_ai_response(response_text: str, expected_format: str = "json",
                        required_keys: List[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate AI response format and content.

    Args:
        response_text: Raw AI response text
        expected_format: Expected response format ("json", "text", "structured")
        required_keys: Required keys for JSON responses

    Returns:
        Tuple of (is_valid, parsed_response)
    """
    try:
        validation_result = {"is_valid": False, "parsed_data": None, "errors": []}

        if not response_text or not response_text.strip():
            validation_result["errors"].append("Empty response received")
            return False, validation_result

        # Clean the response text
        cleaned_text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', response_text)

        if expected_format == "json":
            # Try to extract and parse JSON
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", cleaned_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{[\s\S]*\}', cleaned_text)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    validation_result["errors"].append("No JSON found in response")
                    return False, validation_result

            # Clean JSON string
            json_str = re.sub(r"//.*", "", json_str)  # Remove comments
            json_str = re.sub(r",\s*}", "}", json_str)  # Remove trailing commas
            json_str = re.sub(r",\s*]", "]", json_str)

            try:
                parsed_json = json.loads(json_str)
                validation_result["parsed_data"] = parsed_json

                # Check required keys
                if required_keys:
                    missing_keys = []
                    for key in required_keys:
                        if key not in parsed_json:
                            missing_keys.append(key)

                    if missing_keys:
                        validation_result["errors"].append(f"Missing required keys: {missing_keys}")
                        return False, validation_result

                validation_result["is_valid"] = True
                logger.debug("AI response validation successful")
                return True, validation_result

            except json.JSONDecodeError as e:
                validation_result["errors"].append(f"JSON parsing error: {e}")
                return False, validation_result

        elif expected_format == "text":
            # Simple text validation
            if len(cleaned_text.strip()) > 10:  # Minimum meaningful length
                validation_result["is_valid"] = True
                validation_result["parsed_data"] = cleaned_text.strip()
                return True, validation_result
            else:
                validation_result["errors"].append("Response too short to be meaningful")
                return False, validation_result

        elif expected_format == "structured":
            # Check for structured content (headers, lists, etc.)
            has_structure = bool(
                re.search(r'^\s*[-*+]\s+', cleaned_text, re.MULTILINE) or  # Lists
                re.search(r'^\s*\d+\.\s+', cleaned_text, re.MULTILINE) or  # Numbered lists
                re.search(r'^#+\s+', cleaned_text, re.MULTILINE) or  # Headers
                re.search(r'\*\*.*?\*\*', cleaned_text)  # Bold text
            )

            if has_structure and len(cleaned_text.strip()) > 50:
                validation_result["is_valid"] = True
                validation_result["parsed_data"] = cleaned_text.strip()
                return True, validation_result
            else:
                validation_result["errors"].append("Response lacks expected structure")
                return False, validation_result

        validation_result["errors"].append(f"Unknown expected format: {expected_format}")
        return False, validation_result

    except Exception as e:
        logger.error(f"Error validating AI response: {e}")
        return False, {"is_valid": False, "parsed_data": None, "errors": [str(e)]}


def get_ai_model_status(analyzer) -> Dict[str, Any]:
    """
    Get current AI model status and availability.

    Args:
        analyzer: GeminiAnalyzer instance

    Returns:
        Dictionary containing model status information
    """
    try:
        status = {
            "api_key_valid": getattr(analyzer, 'api_key_valid', False),
            "current_priority_list": getattr(analyzer, 'model_priority_list', []),
            "last_successful_model": None,
            "total_api_calls": getattr(analyzer, 'total_api_calls', 0),
            "failed_calls": getattr(analyzer, 'failed_calls', 0),
            "success_rate": 0.0
        }

        # Calculate success rate
        if status["total_api_calls"] > 0:
            status["success_rate"] = (status["total_api_calls"] - status["failed_calls"]) / status["total_api_calls"]

        # Get timer information if available
        if hasattr(analyzer, 'timer') and analyzer.timer:
            timer_stats = analyzer.timer.get_statistics()
            status["api_timing"] = {
                "total_time": timer_stats.get("total_time", 0),
                "average_time": timer_stats.get("average_time", 0),
                "call_count": timer_stats.get("call_count", 0)
            }

        logger.debug(f"AI model status: {status}")
        return status

    except Exception as e:
        logger.error(f"Error getting AI model status: {e}")
        return {"error": str(e)}


def test_ai_model_connectivity(analyzer, test_prompt: str = "Hello, please respond with 'OK'") -> Dict[str, Any]:
    """
    Test AI model connectivity and response quality.

    Args:
        analyzer: GeminiAnalyzer instance
        test_prompt: Simple test prompt

    Returns:
        Dictionary containing connectivity test results
    """
    try:
        if not analyzer.api_key_valid:
            return {
                "success": False,
                "error": "API key not valid",
                "models_tested": [],
                "response_time": 0
            }

        import time
        start_time = time.time()

        # Test with a simple prompt
        response = analyzer._call_gemini(test_prompt)

        end_time = time.time()
        response_time = end_time - start_time

        # Validate response
        is_valid, validation_result = validate_ai_response(response, "text")

        result = {
            "success": is_valid,
            "response_time": response_time,
            "response_length": len(response) if response else 0,
            "models_tested": analyzer.model_priority_list,
            "validation_result": validation_result
        }

        if not is_valid:
            result["error"] = "Invalid response received"

        logger.info(f"AI connectivity test completed: {result['success']}")
        return result

    except Exception as e:
        logger.error(f"Error testing AI model connectivity: {e}")
        return {
            "success": False,
            "error": str(e),
            "models_tested": [],
            "response_time": 0
        }


def _create_historical_performance_summary_for_ai(cycle_history: List[Dict]) -> str:
    """
    Creates a concise, natural language summary of the run's history for the AI to analyze,
    now including detailed drawdown context and the specific features used.
    """
    if not cycle_history:
        return "No performance history yet. This is the first cycle."

    summary = "### Walk-Forward Performance Summary ###\n\n"
    for i, cycle in enumerate(cycle_history):
        metrics = cycle.get('metrics', {})
        status = cycle.get('status', 'Unknown')

        cycle_summary = f"Cycle {i+1} ({status}): "

        if metrics:
            mar_ratio = metrics.get('mar_ratio', 0)
            total_trades = metrics.get('total_trades', 0)
            max_dd = metrics.get('max_drawdown_pct', 0)
            win_rate = metrics.get('win_rate', 0)

            cycle_summary += f"MAR={mar_ratio:.2f}, Trades={total_trades}, MaxDD={max_dd:.1%}, WinRate={win_rate:.1%}"
        else:
            cycle_summary += "No metrics available"

        summary += cycle_summary + "\n"

    # Add overall trend analysis
    if len(cycle_history) >= 2:
        recent_mars = [c.get('metrics', {}).get('mar_ratio', 0) for c in cycle_history[-3:]]
        recent_mars = [mar for mar in recent_mars if mar is not None]

        if len(recent_mars) >= 2:
            if recent_mars[-1] > recent_mars[-2]:
                summary += "\nTrend: Performance improving in recent cycles."
            elif recent_mars[-1] < recent_mars[-2]:
                summary += "\nTrend: Performance declining in recent cycles."
            else:
                summary += "\nTrend: Performance stable in recent cycles."

    return summary


def _create_label_distribution_report(df: pd.DataFrame, target_col: str) -> str:
    """
    Generates a report on the class balance of the labels for the AI.
    """
    if target_col not in df.columns or df[target_col].isnull().all():
        return f"Label Distribution Report: Target column '{target_col}' not found or is all NaN."

    counts = df[target_col].value_counts(normalize=True)
    # Map numeric labels to meaningful names
    label_map = {0: 'Short', 1: 'Hold', 2: 'Long'}
    report_dict = {label_map.get(k, k): f"{v:.2%}" for k, v in counts.to_dict().items()}

    return f"Label Distribution for '{target_col}': {report_dict}"


def _create_optuna_summary_for_ai(study, top_n: int = 3) -> str:
    """
    Creates a concise summary of the top Optuna trials for the AI, compatible with multi-objective studies.
    """
    if not study or not study.trials:
        return "Optuna Summary: No trials were completed."

    summary = "Optuna Summary (Top Trials sorted by Objective 1):\n"

    # Filter out pruned or failed trials
    try:
        import optuna
        successful_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    except ImportError:
        successful_trials = [t for t in study.trials if hasattr(t, 'values') and t.values is not None]

    if not successful_trials:
        return "Optuna Summary: No successful trials completed."

    # Sort by first objective (assuming higher is better)
    successful_trials.sort(key=lambda t: t.values[0] if t.values else -float('inf'), reverse=True)

    for i, trial in enumerate(successful_trials[:top_n]):
        if hasattr(trial, 'values') and trial.values:
            if len(trial.values) == 1:
                summary += f"  Trial {trial.number}: Score={trial.values[0]:.4f}, Params={trial.params}\n"
            else:
                summary += f"  Trial {trial.number}: Scores={[round(v, 4) for v in trial.values]}, Params={trial.params}\n"
        else:
            summary += f"  Trial {trial.number}: No values available\n"

    return summary
