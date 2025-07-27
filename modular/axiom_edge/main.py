#!/usr/bin/env python3
"""
AxiomEdge Professional Trading Framework - Main Entry Point

This module serves as the primary entry point for the AxiomEdge trading framework.
It provides a clean interface for launching the framework with various execution modes
and configuration options.

Usage:
    python -m axiom_edge.main                    # Run with default settings
    python -m axiom_edge.main --config config.json  # Run with custom config
    python -m axiom_edge.main --daemon --hours 24    # Run in daemon mode
    python -m axiom_edge.main --validate             # Validate installation
    python -m axiom_edge.main --info                 # Show framework info

Features:
    - Single-run and daemon execution modes
    - Custom configuration file support
    - Installation validation and diagnostics
    - Framework information and capabilities
    - Professional logging and error handling
    - Graceful shutdown and cleanup
"""

import sys
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

# Import framework components
from . import (
    __version__, 
    get_framework_info, 
    validate_installation,
    create_framework,
    CAPABILITIES
)
from .tasks import main as framework_main
from .utils import _setup_logging, flush_loggers
from .config import create_default_config

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Setup command line argument parser with comprehensive options.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog='axiom_edge',
        description='AxiomEdge Professional Trading Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run framework with default settings
  %(prog)s --config my_config.json   # Run with custom configuration
  %(prog)s --daemon --hours 24       # Run in daemon mode for 24 hours
  %(prog)s --validate                # Validate installation
  %(prog)s --info                    # Show framework information
  %(prog)s --version                 # Show version information
        """
    )
    
    # Execution modes
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        '--daemon', 
        action='store_true',
        help='Run in daemon mode for continuous execution'
    )
    execution_group.add_argument(
        '--validate', 
        action='store_true',
        help='Validate installation and dependencies'
    )
    execution_group.add_argument(
        '--info', 
        action='store_true',
        help='Show framework information and capabilities'
    )
    execution_group.add_argument(
        '--version', 
        action='version',
        version=f'AxiomEdge v{__version__}'
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        metavar='FILE',
        help='Path to configuration file (JSON format)'
    )
    
    # Daemon mode options
    parser.add_argument(
        '--hours',
        type=int,
        default=0,
        metavar='N',
        help='Run daemon for N hours (0 = unlimited, requires --daemon)'
    )
    
    parser.add_argument(
        '--max-runs',
        type=int,
        default=1,
        metavar='N',
        help='Maximum number of runs (1 = single run, >1 = multiple runs)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        metavar='FILE',
        help='Path to log file (default: auto-generated)'
    )
    
    # Data directory
    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        metavar='DIR',
        help='Directory containing data files (default: current directory)'
    )
    
    # Advanced options
    parser.add_argument(
        '--no-telemetry',
        action='store_true',
        help='Disable telemetry collection'
    )
    
    parser.add_argument(
        '--api-interval',
        type=int,
        default=61,
        metavar='SECONDS',
        help='API call interval in seconds (default: 61)'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments for consistency and correctness.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if arguments are valid, False otherwise
    """
    # Validate daemon mode options
    if args.hours > 0 and not args.daemon:
        print("Error: --hours option requires --daemon mode")
        return False
    
    # Validate configuration file
    if args.config and not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        return False
    
    # Validate data directory
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return False
    
    # Validate log file directory
    if args.log_file:
        log_dir = Path(args.log_file).parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error: Cannot create log directory: {e}")
                return False
    
    return True


def show_framework_info():
    """Display comprehensive framework information."""
    info = get_framework_info()
    
    print(f"🚀 AxiomEdge Professional Trading Framework v{__version__}")
    print("=" * 60)
    print(f"📊 Components: {len(info['components'])}")
    for component in info['components']:
        print(f"   • {component}")
    
    print(f"\n⚡ Capabilities: {len(info['capabilities'])}")
    for capability in info['capabilities']:
        print(f"   • {capability}")
    
    print(f"\n🔧 Available Features:")
    for feature, available in info['capabilities_available'].items():
        status = "✅" if available else "❌"
        print(f"   {status} {feature.replace('_', ' ').title()}")
    
    print(f"\n🐍 Python Version: {info['python_version']}")
    print(f"📦 Total Exports: {info['total_exports']}")
    print(f"📅 Build Date: {info['build_date']}")


def show_validation_results():
    """Display installation validation results."""
    results = validate_installation()
    
    print("🔍 AxiomEdge Installation Validation")
    print("=" * 40)
    
    # Overall status
    if results['installation_valid']:
        print("✅ Installation: VALID")
    else:
        print("❌ Installation: INVALID")
    
    # Python version
    if results['python_version_ok']:
        print("✅ Python Version: OK")
    else:
        print("❌ Python Version: INCOMPATIBLE")
    
    # Core modules
    if results['core_modules_available']:
        print("✅ Core Modules: AVAILABLE")
    else:
        print("❌ Core Modules: MISSING")
    
    # Optional modules
    print("\n📦 Optional Modules:")
    for module, available in results['optional_modules'].items():
        status = "✅" if available else "❌"
        print(f"   {status} {module.replace('_', ' ').title()}")
    
    # Warnings
    if results['warnings']:
        print("\n⚠️  Warnings:")
        for warning in results['warnings']:
            print(f"   • {warning}")
    
    # Errors
    if results['errors']:
        print("\n❌ Errors:")
        for error in results['errors']:
            print(f"   • {error}")
    
    return results['installation_valid']


def validate_environment():
    """Validate environment variables and framework configuration."""
    issues = []
    recommendations = []

    # Check Gemini API key (only required API key)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key or "YOUR" in gemini_key or "PASTE" in gemini_key:
        issues.append("GEMINI_API_KEY not configured")
        recommendations.append("Set GEMINI_API_KEY for AI analysis and enhanced data collection")
        recommendations.append("Get free API key at: https://makersuite.google.com/app/apikey")

    # Framework is self-sufficient for financial data
    print("🚀 AxiomEdge Framework Status:")
    print("   ✅ Self-sufficient financial data collection")
    print("   ✅ Yahoo Finance integration (no API key required)")
    print("   ✅ Sample data generation for testing")
    print("   ✅ CSV file support for custom data")

    if gemini_key and "YOUR" not in gemini_key and "PASTE" not in gemini_key:
        print("   ✅ Gemini AI integration for enhanced data search")
    else:
        print("   ⚠️  Gemini AI not configured (optional but recommended)")

    if issues:
        print("\n💡 Optional Enhancements:")
        for rec in recommendations:
            print(f"   • {rec}")
        print()
    else:
        print("\n✅ Framework is ready to use!")
        print()


def main():
    """
    Main entry point for the AxiomEdge framework.
    
    Handles command line arguments, validates installation, and launches
    the appropriate framework execution mode.
    """
    # Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Handle special modes
    if args.info:
        show_framework_info()
        return
    
    if args.validate:
        if show_validation_results():
            print("\n✅ Installation validation passed!")
            sys.exit(0)
        else:
            print("\n❌ Installation validation failed!")
            sys.exit(1)

    # Validate environment configuration
    validate_environment()
    
    # Setup logging
    if args.log_file:
        log_file = args.log_file
    else:
        from datetime import datetime
        log_file = f"axiom_edge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    _setup_logging(log_file, f"AxiomEdge_v{__version__}")
    logger.info(f"AxiomEdge v{__version__} starting...")
    logger.info(f"Command line args: {vars(args)}")
    
    try:
        # Validate installation before running
        validation_results = validate_installation()
        if not validation_results['installation_valid']:
            logger.error("Installation validation failed")
            print("❌ Installation validation failed. Use --validate for details.")
            sys.exit(1)
        
        # Change to data directory
        original_cwd = Path.cwd()
        data_path = Path(args.data_dir).resolve()
        
        logger.info(f"Changing to data directory: {data_path}")
        import os
        os.chdir(data_path)
        
        # Run the framework
        logger.info("Launching AxiomEdge framework...")
        framework_main()
        
        logger.info("AxiomEdge framework completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Framework interrupted by user")
        print("\n🛑 Framework stopped by user")
        
    except Exception as e:
        logger.error(f"Framework execution failed: {e}", exc_info=True)
        print(f"❌ Framework execution failed: {e}")
        sys.exit(1)
        
    finally:
        # Restore original directory
        try:
            os.chdir(original_cwd)
        except:
            pass
        
        # Flush loggers
        flush_loggers()


if __name__ == '__main__':
    main()
