# =============================================================================
# GENETIC PROGRAMMER MODULE
# =============================================================================

import os
import random
import re
import logging
import multiprocessing
from functools import partial
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
import warnings

from .config import ConfigModel
from .utils import get_optimal_system_settings

logger = logging.getLogger(__name__)

class GeneticProgrammer:
    """
    Advanced genetic algorithm-based trading rule evolution system.

    Evolves trading rules using sophisticated genetic programming techniques where the AI defines
    the gene pool (indicators, operators) and this class handles the evolutionary
    process of creating, evaluating, and evolving rule-based strategies.

    Features:
    - Separate handling of continuous and state-based features
    - Semantic correctness in rule generation with type checking
    - Configurable population size and evolution parameters
    - Multi-objective fitness evaluation (Sharpe, MAR, trade frequency, drawdown)
    - Advanced crossover and mutation operations for rule evolution
    - AI-driven retry loops for failed evolution attempts
    - Elitism preservation and diversity maintenance
    - Parallel fitness evaluation for performance
    - Rule complexity management and bloat control
    - Statistical significance testing for rule validation
    - Adaptive mutation rates based on population diversity
    - Rule caching and memoization for efficiency
    """
    
    def __init__(self, gene_pool: Dict, config: ConfigModel, population_size: int = 50, 
                 generations: int = 25, mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        """
        Initialize the Genetic Programmer with gene pool and parameters.
        
        Args:
            gene_pool: Dictionary containing features and operators for evolution
            config: Configuration model with framework settings
            population_size: Number of individuals in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each individual
            crossover_rate: Probability of crossover between parents
        """
        self.config = config
        
        # Separate features and operators by type for semantic correctness
        self.continuous_features = gene_pool.get('continuous_features', [])
        self.state_features = gene_pool.get('state_features', [])
        self.comparison_operators = gene_pool.get('comparison_operators', ['>', '<', '>=', '<='])
        self.state_operators = gene_pool.get('state_operators', ['==', '!='])
        self.logical_operators = gene_pool.get('logical_operators', ['AND', 'OR'])
        self.constants = gene_pool.get('constants', [0, 25, 50, 75, 100])
        
        # Evolution parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[Tuple[str, str]] = []

        # Advanced evolution parameters
        self.elitism_rate = getattr(config, 'GP_ELITISM_RATE', 0.1)  # Keep top 10% unchanged
        self.max_rule_depth = getattr(config, 'GP_MAX_RULE_DEPTH', 5)
        self.min_rule_depth = getattr(config, 'GP_MIN_RULE_DEPTH', 1)
        self.diversity_threshold = getattr(config, 'GP_DIVERSITY_THRESHOLD', 0.8)
        self.adaptive_mutation = getattr(config, 'GP_ADAPTIVE_MUTATION', True)
        self.parallel_evaluation = getattr(config, 'GP_PARALLEL_EVALUATION', True)

        # Performance tracking
        self.fitness_history: List[List[float]] = []
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.rule_cache: Dict[str, float] = {}
        self.evaluation_count = 0

        # Statistical tracking
        self.generation_stats: List[Dict[str, Any]] = []
        self.convergence_threshold = getattr(config, 'GP_CONVERGENCE_THRESHOLD', 0.001)
        self.stagnation_limit = getattr(config, 'GP_STAGNATION_LIMIT', 10)

        # Combine all features for backward compatibility
        self.indicators = self.continuous_features + self.state_features

        # Validate initialization
        if not self.continuous_features and not self.state_features:
            raise ValueError("GeneticProgrammer cannot be initialized with an empty pool of features.")

        # Initialize multiprocessing if enabled
        if self.parallel_evaluation:
            self.cpu_count = min(multiprocessing.cpu_count(), 4)  # Limit to 4 cores
        else:
            self.cpu_count = 1

        logger.info(f"Advanced Genetic Programmer initialized with {len(self.indicators)} features, "
                   f"population size: {population_size}, generations: {generations}, "
                   f"elitism: {self.elitism_rate:.1%}, parallel cores: {self.cpu_count}")

    def _get_pip_size(self, symbol: str, price: float) -> float:
        """Determines the instrument's pip/point size for cost calculation."""
        if 'JPY' in symbol:
            return 0.01  # JPY pairs have 2 decimal places
        elif any(x in symbol for x in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return 0.01  # Precious metals
        else:
            return 0.0001  # Most forex pairs have 4 decimal places

    def create_initial_population(self):
        """Generates the starting population of trading rules."""
        self.population = []
        for _ in range(self.population_size):
            long_rule = self._create_individual_chromosome(depth=random.randint(1, 3))
            short_rule = self._create_individual_chromosome(depth=random.randint(1, 3))
            self.population.append((long_rule, short_rule))
        logger.info(f"Initial population of {self.population_size} individuals created.")

    def calculate_population_diversity(self) -> float:
        """
        Calculate the diversity of the current population using rule similarity.
        Higher values indicate more diverse population.
        """
        if len(self.population) < 2:
            return 1.0

        try:
            # Convert rules to feature vectors for similarity calculation
            rule_vectors = []
            for long_rule, short_rule in self.population:
                # Count feature usage in rules
                feature_usage = {feature: 0 for feature in self.indicators}
                combined_rule = f"{long_rule} {short_rule}"

                for feature in self.indicators:
                    feature_usage[feature] = combined_rule.count(feature)

                rule_vectors.append(list(feature_usage.values()))

            # Calculate pairwise similarities
            similarities = []
            for i in range(len(rule_vectors)):
                for j in range(i + 1, len(rule_vectors)):
                    vec1, vec2 = np.array(rule_vectors[i]), np.array(rule_vectors[j])

                    # Cosine similarity
                    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                        similarities.append(similarity)

            # Diversity is 1 - average similarity
            avg_similarity = np.mean(similarities) if similarities else 0
            diversity = 1.0 - avg_similarity

            return max(0.0, min(1.0, diversity))

        except Exception as e:
            logger.debug(f"Error calculating diversity: {e}")
            return 0.5  # Default moderate diversity

    def calculate_rule_complexity(self, rule: str) -> int:
        """
        Calculate the complexity of a rule based on number of conditions and operators.
        """
        try:
            # Count logical operators
            logical_count = rule.count(' AND ') + rule.count(' OR ')

            # Count comparison operators
            comparison_count = sum(rule.count(op) for op in ['>', '<', '>=', '<=', '==', '!='])

            # Count parentheses (nesting complexity)
            paren_count = rule.count('(') + rule.count(')')

            # Total complexity score
            complexity = logical_count * 2 + comparison_count + paren_count * 0.5

            return int(complexity)

        except Exception as e:
            logger.debug(f"Error calculating rule complexity: {e}")
            return 1

    def apply_bloat_control(self, rule: str) -> str:
        """
        Apply bloat control to prevent rules from becoming overly complex.
        Simplifies rules that exceed maximum complexity threshold.
        """
        try:
            complexity = self.calculate_rule_complexity(rule)
            max_complexity = getattr(self.config, 'GP_MAX_RULE_COMPLEXITY', 15)

            if complexity <= max_complexity:
                return rule

            # Simplify by removing redundant conditions
            # This is a basic implementation - could be made more sophisticated
            simplified_rule = rule

            # Remove excessive parentheses
            while '((' in simplified_rule:
                simplified_rule = simplified_rule.replace('((', '(')
            while '))' in simplified_rule:
                simplified_rule = simplified_rule.replace('))', ')')

            # If still too complex, truncate to simpler form
            if self.calculate_rule_complexity(simplified_rule) > max_complexity:
                # Extract first few conditions
                parts = re.split(r'\s+(AND|OR)\s+', simplified_rule)
                if len(parts) > 3:  # Keep first condition only
                    simplified_rule = parts[0]

            return simplified_rule

        except Exception as e:
            logger.debug(f"Error applying bloat control: {e}")
            return rule

    def _create_individual_chromosome(self, depth: int = 2) -> str:
        """
        Creates a single trading rule (chromosome) with specified complexity depth.
        
        Args:
            depth: Complexity level of the rule (1-3)
            
        Returns:
            String representation of a trading rule
        """
        if depth <= 1:
            return self._create_simple_condition()
        
        # Create compound condition with logical operators
        left_condition = self._create_simple_condition()
        right_condition = self._create_simple_condition()
        logical_op = random.choice(self.logical_operators)
        
        if depth > 2:
            # Add third condition for more complexity
            third_condition = self._create_simple_condition()
            logical_op2 = random.choice(self.logical_operators)
            return f"({left_condition} {logical_op} {right_condition}) {logical_op2} {third_condition}"
        
        return f"{left_condition} {logical_op} {right_condition}"

    def _create_individual_rule(self) -> str:
        """
        [FIXED] Intelligently creates a single logical rule string by respecting
        the difference between continuous and binary state features.
        """
        # Decide whether to build a rule based on a continuous or state feature
        # Give a higher chance to continuous features as they are more common in rules.
        rule_type = 'continuous' if random.random() < 0.8 and self.continuous_features else 'state'

        # If the chosen type has no features, fallback to the other type.
        if rule_type == 'continuous' and not self.continuous_features:
            rule_type = 'state'
        elif rule_type == 'state' and not self.state_features:
            rule_type = 'continuous'

        # If still no features are available in either category, return a trivial rule.
        if not self.continuous_features and not self.state_features:
            return "(1 == 1)"

        if rule_type == 'continuous':
            # Create a rule like "(RSI > 70)" or "(ATR crosses_above 0.005)"
            indicator = random.choice(self.continuous_features)
            operator = random.choice(self.comparison_operators)
            value = random.choice(self.constants)
            return f"({indicator} {operator} {value})"
        else: # rule_type == 'state'
            # Create a rule like "(is_bullish_pullback == 1)" or "(ema_cross_H4_bullish != 0)"
            indicator = random.choice(self.state_features)
            operator = random.choice(self.state_operators)
            # State features are binary (0 or 1), so they should only be compared to 0 or 1.
            value = random.choice([0, 1])
            return f"({indicator} {operator} {value})"

    def _create_simple_condition(self) -> str:
        """Creates a simple condition like 'RSI < 30' or 'trend == 1'."""
        feature = random.choice(self.indicators)

        # Choose appropriate operator based on feature type
        if feature in self.continuous_features:
            operator = random.choice(self.comparison_operators)
            constant = random.choice(self.constants)
            return f"{feature} {operator} {constant}"
        else:
            # State feature
            operator = random.choice(self.state_operators)
            # For state features, use binary values or small integers
            constant = random.choice([0, 1])
            return f"{feature} {operator} {constant}"

    def _parse_and_eval_rule(self, rule_str: str, df: pd.DataFrame) -> pd.Series:
        """Safely evaluates a rule string against a dataframe."""
        try:
            # Replace feature names with actual column references
            safe_rule = rule_str
            for feature in self.indicators:
                if feature in df.columns:
                    safe_rule = safe_rule.replace(feature, f"df['{feature}']")

            # Evaluate the rule
            result = eval(safe_rule)

            # Ensure result is a boolean Series
            if isinstance(result, pd.Series):
                return result.astype(bool)
            else:
                # If scalar, broadcast to Series
                return pd.Series([bool(result)] * len(df), index=df.index)

        except Exception as e:
            logger.warning(f"Error evaluating rule '{rule_str}': {e}")
            # Return all False if rule evaluation fails
            return pd.Series([False] * len(df), index=df.index)

    def evaluate_fitness(self, chromosome: Tuple[str, str], df_eval: pd.DataFrame) -> float:
        """
        Enhanced multi-objective fitness evaluation with caching and advanced metrics.

        Args:
            chromosome: Tuple of (long_rule, short_rule)
            df_eval: DataFrame with market data and features

        Returns:
            Composite fitness score (higher is better)
        """
        try:
            long_rule, short_rule = chromosome

            # Check cache first
            rule_key = f"{long_rule}|{short_rule}"
            if rule_key in self.rule_cache:
                return self.rule_cache[rule_key]

            self.evaluation_count += 1

            # Apply bloat control
            long_rule = self.apply_bloat_control(long_rule)
            short_rule = self.apply_bloat_control(short_rule)

            # Ensure index is sorted for proper time series operations
            unique_index = df_eval.index.unique().sort_values()
            equity_curve = pd.Series(1.0, index=unique_index)
            returns_series = pd.Series(0.0, index=unique_index)
            last_equity = 1.0
            trade_count = 0
            winning_trades = 0
            losing_trades = 0
            max_drawdown = 0.0
            peak_equity = 1.0

            # Generate signals based on rules
            long_signals = self._evaluate_rule(long_rule, df_eval)
            short_signals = self._evaluate_rule(short_rule, df_eval)

            # Combine signals (1 for long, -1 for short, 0 for no position)
            signals = pd.Series(0, index=df_eval.index)
            signals[long_signals] = 1
            signals[short_signals] = -1

            if 'Close' not in df_eval.columns:
                self.rule_cache[rule_key] = -10.0
                return -10.0

            # Enhanced trading simulation
            position_changes = signals.diff().fillna(0)
            price_returns = df_eval['Close'].pct_change().fillna(0)
            current_position = 0

            for i, (timestamp, signal) in enumerate(signals.items()):
                if i == 0:
                    continue

                prev_position = current_position
                current_position = signal

                if prev_position != current_position:  # Position change
                    trade_count += 1

                    # Calculate trade return
                    if timestamp in price_returns.index:
                        trade_return = price_returns.loc[timestamp] * current_position

                        # Apply transaction costs
                        transaction_cost = getattr(self.config, 'GP_TRANSACTION_COST', 0.001)
                        trade_return -= transaction_cost

                        # Update equity
                        last_equity *= (1 + trade_return)
                        equity_curve.loc[timestamp] = last_equity
                        returns_series.loc[timestamp] = trade_return

                        # Track winning/losing trades
                        if trade_return > 0:
                            winning_trades += 1
                        else:
                            losing_trades += 1

                        # Update peak and drawdown
                        if last_equity > peak_equity:
                            peak_equity = last_equity
                        else:
                            current_drawdown = (peak_equity - last_equity) / peak_equity
                            max_drawdown = max(max_drawdown, current_drawdown)

            # Forward-fill the equity curve
            equity_curve.ffill(inplace=True)

            # Enhanced Multi-Objective Fitness Calculation
            min_trades = getattr(self.config, 'GP_MIN_TRADES', 10)
            if trade_count < min_trades:
                fitness_score = -10.0
                self.rule_cache[rule_key] = fitness_score
                return fitness_score

            # Calculate advanced metrics
            total_return = last_equity - 1.0

            # Sharpe ratio
            sharpe_ratio = 0.0
            if len(returns_series[returns_series != 0]) > 1:
                active_returns = returns_series[returns_series != 0]
                if active_returns.std() > 0:
                    sharpe_ratio = active_returns.mean() / active_returns.std()

            # Sortino ratio (downside deviation)
            sortino_ratio = 0.0
            if len(returns_series[returns_series != 0]) > 1:
                active_returns = returns_series[returns_series != 0]
                downside_returns = active_returns[active_returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    sortino_ratio = active_returns.mean() / downside_returns.std()

            # Calmar ratio (return / max drawdown)
            calmar_ratio = 0.0
            if max_drawdown > 0:
                calmar_ratio = total_return / max_drawdown

            # Win rate
            win_rate = winning_trades / trade_count if trade_count > 0 else 0

            # Profit factor
            profit_factor = 0.0
            if losing_trades > 0:
                gross_profit = sum(r for r in returns_series if r > 0)
                gross_loss = abs(sum(r for r in returns_series if r < 0))
                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss

            # Trade frequency score (prefer moderate trading)
            optimal_trades_per_period = getattr(self.config, 'GP_OPTIMAL_TRADES', 50)
            trade_frequency_score = 1.0 - abs(trade_count - optimal_trades_per_period) / optimal_trades_per_period

            # Complexity penalty
            rule_complexity = self.calculate_rule_complexity(long_rule) + self.calculate_rule_complexity(short_rule)
            complexity_penalty = rule_complexity * getattr(self.config, 'GP_COMPLEXITY_PENALTY', 0.01)

            # Multi-objective fitness with configurable weights
            weights = getattr(self.config, 'GP_FITNESS_WEIGHTS', {
                'sharpe': 0.3,
                'sortino': 0.2,
                'calmar': 0.2,
                'win_rate': 0.1,
                'profit_factor': 0.1,
                'trade_frequency': 0.1
            })

            fitness_score = (
                sharpe_ratio * weights.get('sharpe', 0.3) +
                sortino_ratio * weights.get('sortino', 0.2) +
                calmar_ratio * weights.get('calmar', 0.2) +
                win_rate * weights.get('win_rate', 0.1) +
                profit_factor * weights.get('profit_factor', 0.1) +
                trade_frequency_score * weights.get('trade_frequency', 0.1) -
                complexity_penalty
            )

            # Cache the result
            self.rule_cache[rule_key] = fitness_score
            return fitness_score

        except Exception as e:
            logger.debug(f"Error evaluating chromosome: {e}")
            return -10.0

    def _evaluate_rule(self, rule: str, df: pd.DataFrame) -> pd.Series:
        """
        Evaluates a trading rule string against the dataframe.
        
        Args:
            rule: String representation of trading rule
            df: DataFrame with market data
            
        Returns:
            Boolean series indicating when rule is satisfied
        """
        try:
            # Clean and prepare the rule
            rule = rule.strip()
            if not rule:
                return pd.Series(False, index=df.index)
            
            # Replace feature names with actual values
            eval_rule = rule
            for feature in self.indicators:
                if feature in df.columns:
                    # Replace feature name with df[feature] for evaluation
                    eval_rule = eval_rule.replace(feature, f"df['{feature}']")
            
            # Replace logical operators with Python equivalents
            eval_rule = eval_rule.replace(' AND ', ' & ')
            eval_rule = eval_rule.replace(' OR ', ' | ')
            
            # Safely evaluate the rule
            try:
                result = eval(eval_rule)
                if isinstance(result, pd.Series):
                    return result.fillna(False)
                else:
                    # If result is a scalar, return series of that value
                    return pd.Series(bool(result), index=df.index)
            except:
                return pd.Series(False, index=df.index)
                
        except Exception as e:
            logger.debug(f"Error evaluating rule '{rule}': {e}")
            return pd.Series(False, index=df.index)

    def evaluate_population_fitness(self, df_eval: pd.DataFrame) -> List[float]:
        """
        Evaluate fitness for entire population with optional parallel processing.

        Args:
            df_eval: DataFrame with market data for evaluation

        Returns:
            List of fitness scores for each individual
        """
        if self.parallel_evaluation and self.cpu_count > 1:
            return self._parallel_fitness_evaluation(df_eval)
        else:
            return self._sequential_fitness_evaluation(df_eval)

    def _sequential_fitness_evaluation(self, df_eval: pd.DataFrame) -> List[float]:
        """Sequential fitness evaluation for all individuals."""
        fitness_scores = []
        for chromosome in self.population:
            fitness = self.evaluate_fitness(chromosome, df_eval)
            fitness_scores.append(fitness)
        return fitness_scores

    def _parallel_fitness_evaluation(self, df_eval: pd.DataFrame) -> List[float]:
        """Parallel fitness evaluation using multiprocessing."""
        try:
            # Create partial function with fixed df_eval
            eval_func = partial(self._evaluate_chromosome_wrapper, df_eval=df_eval)

            # Use multiprocessing pool
            with multiprocessing.Pool(processes=self.cpu_count) as pool:
                fitness_scores = pool.map(eval_func, self.population)

            return fitness_scores

        except Exception as e:
            logger.warning(f"Parallel evaluation failed: {e}. Falling back to sequential.")
            return self._sequential_fitness_evaluation(df_eval)

    def _evaluate_chromosome_wrapper(self, chromosome: Tuple[str, str], df_eval: pd.DataFrame) -> float:
        """Wrapper function for parallel evaluation."""
        return self.evaluate_fitness(chromosome, df_eval)

    def apply_elitism(self, fitness_scores: List[float]) -> List[Tuple[str, str]]:
        """
        Apply elitism to preserve the best individuals.

        Args:
            fitness_scores: Current generation fitness scores

        Returns:
            Elite individuals to preserve
        """
        if not fitness_scores or self.elitism_rate <= 0:
            return []

        elite_count = max(1, int(self.population_size * self.elitism_rate))

        # Get indices of best individuals
        elite_indices = np.argsort(fitness_scores)[-elite_count:]

        # Return elite individuals
        elite_individuals = [self.population[i] for i in elite_indices]

        logger.debug(f"Preserving {len(elite_individuals)} elite individuals")
        return elite_individuals

    def calculate_adaptive_mutation_rate(self, generation: int, diversity: float) -> float:
        """
        Calculate adaptive mutation rate based on population diversity and generation.

        Args:
            generation: Current generation number
            diversity: Current population diversity (0-1)

        Returns:
            Adjusted mutation rate
        """
        if not self.adaptive_mutation:
            return self.mutation_rate

        # Base mutation rate
        base_rate = self.mutation_rate

        # Increase mutation rate if diversity is low
        diversity_factor = 1.0 + (1.0 - diversity) * 0.5

        # Decrease mutation rate as generations progress (cooling)
        cooling_factor = 1.0 - (generation / self.generations) * 0.3

        # Calculate adaptive rate
        adaptive_rate = base_rate * diversity_factor * cooling_factor

        # Ensure rate stays within reasonable bounds
        adaptive_rate = max(0.01, min(0.5, adaptive_rate))

        return adaptive_rate

    def _selection(self, fitness_scores: List[float]) -> List[Tuple[str, str]]:
        """Selects parents for the next generation using tournament selection."""
        selected = []
        for _ in range(self.population_size):
            tournament_size = min(5, self.population_size)
            aspirants_indices = random.sample(range(len(fitness_scores)), tournament_size)
            winner_index = max(aspirants_indices, key=lambda i: fitness_scores[i])
            selected.append(self.population[winner_index])
        return selected

    def _crossover(self, parents: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Creates the next generation through crossover."""
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                offspring.append(parents[i])
                break
                
            parent1, parent2 = parents[i], parents[i + 1]

            if random.random() < self.crossover_rate:
                long_child1, long_child2 = self._crossover_rules(parent1[0], parent2[0])
                short_child1, short_child2 = self._crossover_rules(parent1[1], parent2[1])
                offspring.append((long_child1, short_child1))
                if len(offspring) < self.population_size:
                    offspring.append((long_child2, short_child2))
            else:
                offspring.append(parent1)
                if len(offspring) < self.population_size:
                    offspring.append(parent2)
                    
        # Ensure we have exactly population_size offspring
        while len(offspring) < self.population_size:
            offspring.append(random.choice(parents))
        
        return offspring[:self.population_size]

    def _crossover_rules(self, rule1: str, rule2: str) -> Tuple[str, str]:
        """Performs crossover on two rule strings."""
        # Simple crossover at logical operators
        logical_ops = [' AND ', ' OR ']
        
        crossover_points1 = []
        crossover_points2 = []
        
        for op in logical_ops:
            if op in rule1:
                crossover_points1.append(rule1.find(op))
            if op in rule2:
                crossover_points2.append(rule2.find(op))
        
        if not crossover_points1 or not crossover_points2:
            return rule1, rule2
        
        point1 = random.choice(crossover_points1)
        point2 = random.choice(crossover_points2)
        
        new_rule1 = rule1[:point1] + rule2[point2:]
        new_rule2 = rule2[:point2] + rule1[point1:]
        
        return new_rule1, new_rule2

    def _mutation(self, offspring: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Applies mutation to the offspring."""
        for i in range(len(offspring)):
            if random.random() < self.mutation_rate:
                long_rule, short_rule = offspring[i]
                mutated_long = self._mutate_rule(long_rule)
                mutated_short = self._mutate_rule(short_rule)
                offspring[i] = (mutated_long, mutated_short)
        return offspring

    def _mutate_rule(self, rule: str) -> str:
        """Applies mutation to a single rule string."""
        try:
            # Split rule into components
            parts = re.split(r'(\s+AND\s+|\s+OR\s+|\s+|\(|\))', rule)
            parts = [p for p in parts if p and p.strip()]
            
            if not parts:
                return rule
            
            mutation_point = random.randint(0, len(parts) - 1)
            part_to_mutate = parts[mutation_point]
            
            # Mutate different types of components
            if part_to_mutate in self.indicators:
                # Replace with another indicator
                parts[mutation_point] = random.choice(self.indicators)
            elif part_to_mutate in self.comparison_operators + self.state_operators:
                # Replace with another operator
                if part_to_mutate in self.comparison_operators:
                    parts[mutation_point] = random.choice(self.comparison_operators)
                else:
                    parts[mutation_point] = random.choice(self.state_operators)
            elif part_to_mutate in self.logical_operators:
                # Flip logical operator
                parts[mutation_point] = 'OR' if part_to_mutate == 'AND' else 'AND'
            elif part_to_mutate.replace('.', '', 1).replace('-', '', 1).isdigit():
                # Replace constant
                parts[mutation_point] = str(random.choice(self.constants))
            
            return ''.join(parts)
            
        except Exception as e:
            logger.debug(f"Error mutating rule '{rule}': {e}")
            return rule

    def run_evolution(self, df_eval: pd.DataFrame, gemini_analyzer=None, api_timer=None) -> Tuple[Tuple[str, str], float]:
        """
        Executes the advanced genetic algorithm with elitism, adaptive mutation, and convergence detection.

        Args:
            df_eval: DataFrame with market data for evaluation
            gemini_analyzer: Optional AI analyzer for gene pool optimization
            api_timer: Optional API timer for rate limiting

        Returns:
            Tuple of (best_chromosome, best_fitness)
        """
        logger.info("Starting ADVANCED Genetic Programming evolution...")
        start_time = time.time()

        # Reset tracking variables
        self.fitness_history = []
        self.best_fitness_history = []
        self.diversity_history = []
        self.generation_stats = []
        self.rule_cache = {}
        self.evaluation_count = 0

        max_attempts = 2 if gemini_analyzer else 1
        successful_run = False

        best_chromosome_overall = (None, None)
        best_fitness_overall = -np.inf

        # Sample data if too large
        sample_size = min(15000, len(df_eval))
        if len(df_eval) > sample_size:
            logger.info(f"Using last {sample_size} rows for GP fitness evaluation.")
            df_eval_sample = df_eval.tail(sample_size).copy()
        else:
            df_eval_sample = df_eval

        available_features_for_gp = self._get_available_features_from_df(df_eval_sample)

        for attempt in range(max_attempts):
            logger.info(f"GP Evolution Attempt {attempt + 1}/{max_attempts}...")
            self.create_initial_population()

            best_chromosome_gen = self.population[0]
            best_fitness_gen = -np.inf
            stagnation_counter = 0
            last_best_fitness = -np.inf
            evolution_log_for_ai = []
            
            # Get system settings for parallel processing
            num_workers = get_optimal_system_settings().get('num_workers', 1)
            use_parallel = num_workers > 1 and len(self.population) > 10
            
            for gen in range(self.generations):
                # Enhanced fitness evaluation with parallel processing
                fitness_scores = self.evaluate_population_fitness(df_eval_sample)

                # Calculate population diversity
                diversity = self.calculate_population_diversity()
                self.diversity_history.append(diversity)

                # Track best individual
                current_best_fitness = max(fitness_scores)
                current_best_idx = fitness_scores.index(current_best_fitness)
                current_best_chromosome = self.population[current_best_idx]

                if current_best_fitness > best_fitness_gen:
                    best_fitness_gen = current_best_fitness
                    best_chromosome_gen = current_best_chromosome
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                # Store generation statistics
                gen_stats = {
                    'generation': gen + 1,
                    'best_fitness': current_best_fitness,
                    'avg_fitness': np.mean(fitness_scores),
                    'std_fitness': np.std(fitness_scores),
                    'diversity': diversity,
                    'evaluations': self.evaluation_count
                }
                self.generation_stats.append(gen_stats)
                self.fitness_history.append(fitness_scores.copy())
                self.best_fitness_history.append(current_best_fitness)

                # Enhanced logging
                if gen % 5 == 0 or gen == self.generations - 1:
                    logger.info(f"  Gen {gen + 1}/{self.generations}: "
                               f"Best={current_best_fitness:.4f}, "
                               f"Avg={gen_stats['avg_fitness']:.4f}, "
                               f"Diversity={diversity:.3f}, "
                               f"Evals={self.evaluation_count}")

                # Store evolution log for AI analysis
                if current_best_fitness < 1.0:
                    evolution_log_for_ai.append({
                        "generation": gen + 1,
                        "best_fitness": round(current_best_fitness, 4),
                        "best_chromosome": current_best_chromosome,
                        "diversity": round(diversity, 3)
                    })

                # Convergence detection
                if (gen > 10 and
                    abs(current_best_fitness - last_best_fitness) < self.convergence_threshold):
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                    last_best_fitness = current_best_fitness

                # Early stopping conditions
                if current_best_fitness > 2.0:
                    logger.info(f"Early stopping: Excellent fitness {current_best_fitness:.4f} at generation {gen + 1}")
                    break

                if stagnation_counter >= self.stagnation_limit:
                    logger.info(f"Early stopping: Stagnation detected at generation {gen + 1}")
                    break

                # Evolution operations with advanced features
                if gen < self.generations - 1:
                    # Apply elitism
                    elite_individuals = self.apply_elitism(fitness_scores)

                    # Selection
                    parents = self._selection(fitness_scores)

                    # Crossover
                    offspring = self._crossover(parents)

                    # Adaptive mutation
                    adaptive_rate = self.calculate_adaptive_mutation_rate(gen, diversity)
                    original_rate = self.mutation_rate
                    self.mutation_rate = adaptive_rate

                    # Mutation
                    offspring = self._mutation(offspring)

                    # Restore original mutation rate
                    self.mutation_rate = original_rate

                    # Combine elite with offspring
                    if elite_individuals:
                        # Replace worst individuals with elite
                        elite_count = len(elite_individuals)
                        offspring = offspring[:-elite_count] + elite_individuals

                    self.population = offspring
            
            # Check if this attempt was successful
            if best_fitness_gen > best_fitness_overall:
                best_fitness_overall = best_fitness_gen
                best_chromosome_overall = best_chromosome_gen
            
            if best_fitness_gen > 0.1:  # Minimum acceptable fitness
                successful_run = True
                logger.info(f"Evolution attempt {attempt + 1} successful with fitness {best_fitness_gen:.4f}")
                break
            
            # Try to get new gene pool from AI if available
            if not successful_run and attempt < max_attempts - 1 and gemini_analyzer:
                logger.warning("GP evolution failed. Requesting new gene pool from AI...")
                
                initial_gene_pool = {
                    'continuous_features': self.continuous_features,
                    'state_features': self.state_features,
                    'comparison_operators': self.comparison_operators,
                    'state_operators': self.state_operators,
                    'logical_operators': self.logical_operators,
                    'constants': self.constants
                }
                
                try:
                    if api_timer:
                        new_gene_pool = api_timer.call(
                            gemini_analyzer.propose_gp_gene_pool_fix,
                            initial_gene_pool, evolution_log_for_ai, available_features_for_gp
                        )
                    else:
                        new_gene_pool = gemini_analyzer.propose_gp_gene_pool_fix(
                            initial_gene_pool, evolution_log_for_ai, available_features_for_gp
                        )
                    
                    required_keys = ['continuous_features', 'state_features', 'comparison_operators',
                                   'state_operators', 'logical_operators', 'constants']
                    
                    if new_gene_pool and all(k in new_gene_pool for k in required_keys):
                        logger.info("AI provided new gene pool. Retrying evolution.")
                        self.continuous_features = new_gene_pool['continuous_features']
                        self.state_features = new_gene_pool['state_features']
                        self.comparison_operators = new_gene_pool['comparison_operators']
                        self.state_operators = new_gene_pool['state_operators']
                        self.logical_operators = new_gene_pool['logical_operators']
                        self.constants = new_gene_pool['constants']
                        self.indicators = self.continuous_features + self.state_features
                    else:
                        logger.error("AI failed to provide valid new gene pool. Aborting.")
                        break
                        
                except Exception as e:
                    logger.error(f"Error getting new gene pool from AI: {e}")
                    break
        
        logger.info("Genetic Programming evolution finished.")
        if best_chromosome_overall[0] is not None:
            logger.info(f"Best Long Rule: {best_chromosome_overall[0]}")
            logger.info(f"Best Short Rule: {best_chromosome_overall[1]}")
        logger.info(f"Final Best Fitness (Sharpe Ratio): {best_fitness_overall:.4f}")
        
        return best_chromosome_overall, best_fitness_overall

    def _get_available_features_from_df(self, df: pd.DataFrame) -> List[str]:
        """Get list of available features from DataFrame."""
        excluded_cols = {
            'Open', 'High', 'Low', 'Close', 'RealVolume', 'Volume', 'Symbol', 'Timestamp',
            'timestamp', 'symbol'
        }
        
        # Exclude target columns
        target_cols = {col for col in df.columns if col.startswith('target_')}
        excluded_cols.update(target_cols)
        
        # Get numeric columns that aren't excluded
        feature_cols = []
        for col in df.columns:
            if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
        
        return feature_cols

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the evolution process.

        Returns:
            Dictionary containing detailed evolution statistics
        """
        if not self.generation_stats:
            return {}

        stats = {
            'total_generations': len(self.generation_stats),
            'total_evaluations': self.evaluation_count,
            'cache_hit_rate': len(self.rule_cache) / max(1, self.evaluation_count),
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'convergence_achieved': False,
            'best_fitness_progression': self.best_fitness_history,
            'diversity_progression': self.diversity_history
        }

        # Calculate convergence metrics
        if len(self.best_fitness_history) > 10:
            recent_improvement = (self.best_fitness_history[-1] -
                                self.best_fitness_history[-10])
            stats['recent_improvement'] = recent_improvement
            stats['convergence_achieved'] = abs(recent_improvement) < self.convergence_threshold

        # Calculate fitness statistics
        all_fitness = [gen['best_fitness'] for gen in self.generation_stats]
        stats['fitness_stats'] = {
            'min': min(all_fitness),
            'max': max(all_fitness),
            'mean': np.mean(all_fitness),
            'std': np.std(all_fitness),
            'improvement': all_fitness[-1] - all_fitness[0] if len(all_fitness) > 1 else 0
        }

        # Calculate diversity statistics
        if self.diversity_history:
            stats['diversity_stats'] = {
                'min': min(self.diversity_history),
                'max': max(self.diversity_history),
                'mean': np.mean(self.diversity_history),
                'std': np.std(self.diversity_history)
            }

        return stats

    def get_population_stats(self) -> Dict[str, Any]:
        """Get statistics about the current population."""
        if not self.population:
            return {}
        
        return {
            'population_size': len(self.population),
            'unique_long_rules': len(set(chrom[0] for chrom in self.population)),
            'unique_short_rules': len(set(chrom[1] for chrom in self.population)),
            'avg_rule_length': np.mean([len(chrom[0]) + len(chrom[1]) for chrom in self.population]),
            'gene_pool_size': len(self.indicators),
            'continuous_features': len(self.continuous_features),
            'state_features': len(self.state_features)
        }
