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

from .config import ConfigModel
from .utils import get_optimal_system_settings

logger = logging.getLogger(__name__)

class GeneticProgrammer:
    """
    Evolves trading rules using a genetic algorithm.
    The AI defines the 'gene pool' (indicators, operators), and this class
    handles the evolutionary process of creating, evaluating, and evolving
    a population of rule-based strategies.
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
        
        # Combine all features for backward compatibility
        self.indicators = self.continuous_features + self.state_features
        
        # Validate initialization
        if not self.continuous_features and not self.state_features:
            raise ValueError("GeneticProgrammer cannot be initialized with an empty pool of features.")
        
        logger.info(f"Genetic Programmer initialized with {len(self.indicators)} features, "
                   f"population size: {population_size}, generations: {generations}")

    def create_initial_population(self):
        """Generates the starting population of trading rules."""
        self.population = []
        for _ in range(self.population_size):
            long_rule = self._create_individual_chromosome(depth=random.randint(1, 3))
            short_rule = self._create_individual_chromosome(depth=random.randint(1, 3))
            self.population.append((long_rule, short_rule))
        logger.info(f"Initial population of {self.population_size} individuals created.")

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

    def evaluate_fitness(self, chromosome: Tuple[str, str], df_eval: pd.DataFrame) -> float:
        """
        Evaluates the fitness of a chromosome (trading rule pair) using Sharpe ratio.
        
        Args:
            chromosome: Tuple of (long_rule, short_rule)
            df_eval: DataFrame with market data and features
            
        Returns:
            Fitness score (Sharpe ratio or similar performance metric)
        """
        try:
            long_rule, short_rule = chromosome
            
            # Generate signals based on rules
            long_signals = self._evaluate_rule(long_rule, df_eval)
            short_signals = self._evaluate_rule(short_rule, df_eval)
            
            # Combine signals (1 for long, -1 for short, 0 for no position)
            signals = pd.Series(0, index=df_eval.index)
            signals[long_signals] = 1
            signals[short_signals] = -1
            
            # Calculate returns
            if 'Close' not in df_eval.columns:
                return -1.0  # Invalid data
            
            price_returns = df_eval['Close'].pct_change().fillna(0)
            strategy_returns = signals.shift(1) * price_returns  # Lag signals by 1 period
            
            # Calculate performance metrics
            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return -1.0
            
            # Sharpe ratio (annualized)
            mean_return = strategy_returns.mean() * 252  # Assuming daily data
            std_return = strategy_returns.std() * np.sqrt(252)
            
            if std_return == 0:
                return -1.0
            
            sharpe_ratio = mean_return / std_return
            
            # Add penalty for excessive trading
            trade_frequency = (signals.diff() != 0).sum() / len(signals)
            if trade_frequency > 0.1:  # More than 10% of days
                sharpe_ratio *= (1 - trade_frequency)
            
            return sharpe_ratio
            
        except Exception as e:
            logger.debug(f"Error evaluating chromosome: {e}")
            return -1.0

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
        Executes the full genetic algorithm to find the best trading rule.
        
        Args:
            df_eval: DataFrame with market data for evaluation
            gemini_analyzer: Optional AI analyzer for gene pool optimization
            api_timer: Optional API timer for rate limiting
            
        Returns:
            Tuple of (best_chromosome, best_fitness)
        """
        logger.info("Starting Genetic Programming evolution...")
        
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
            evolution_log_for_ai = []
            
            # Get system settings for parallel processing
            num_workers = get_optimal_system_settings().get('num_workers', 1)
            use_parallel = num_workers > 1 and len(self.population) > 10
            
            for gen in range(self.generations):
                # Evaluate fitness
                if use_parallel:
                    try:
                        with multiprocessing.Pool(processes=num_workers) as pool:
                            eval_func = partial(self.evaluate_fitness, df_eval=df_eval_sample)
                            fitness_scores = pool.map(eval_func, self.population)
                    except Exception as e:
                        logger.warning(f"Parallel processing failed: {e}. Using serial processing.")
                        fitness_scores = [self.evaluate_fitness(chrom, df_eval_sample) for chrom in self.population]
                else:
                    fitness_scores = [self.evaluate_fitness(chrom, df_eval_sample) for chrom in self.population]
                
                # Track best individual
                current_best_fitness = max(fitness_scores)
                current_best_idx = fitness_scores.index(current_best_fitness)
                current_best_chromosome = self.population[current_best_idx]
                
                if current_best_fitness > best_fitness_gen:
                    best_fitness_gen = current_best_fitness
                    best_chromosome_gen = current_best_chromosome
                
                # Log progress
                if gen % 5 == 0 or gen == self.generations - 1:
                    avg_fitness = np.mean(fitness_scores)
                    logger.info(f"  Generation {gen + 1}/{self.generations}: "
                               f"Best={best_fitness_gen:.4f}, Avg={avg_fitness:.4f}")
                
                # Store evolution log for AI analysis
                if best_fitness_gen < 1.0:
                    evolution_log_for_ai.append({
                        "generation": gen + 1,
                        "best_fitness": round(best_fitness_gen, 4),
                        "best_chromosome": best_chromosome_gen
                    })
                
                # Early stopping if we find a very good solution
                if best_fitness_gen > 2.0:
                    logger.info(f"Early stopping at generation {gen + 1} with fitness {best_fitness_gen:.4f}")
                    break
                
                # Evolution operations
                if gen < self.generations - 1:  # Don't evolve on last generation
                    parents = self._selection(fitness_scores)
                    offspring = self._crossover(parents)
                    self.population = self._mutation(offspring)
            
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
