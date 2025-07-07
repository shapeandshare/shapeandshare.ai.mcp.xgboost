#!/usr/bin/env python3
"""
Cluster Integration Test Runner for MCP XGBoost Application

This script orchestrates comprehensive cluster integration tests using agents
to validate the MCP XGBoost application deployment in Kubernetes.

Features:
- Orchestrates multiple test suites with priority ordering
- Comprehensive reporting and logging
- Prerequisites checking
- Support for different test modes
- JSON and text report generation

Usage:
    python run_cluster_integration_tests.py                    # Run all tests
    python run_cluster_integration_tests.py --suite basic      # Run basic tests only
    python run_cluster_integration_tests.py --quick            # Quick test mode
    python run_cluster_integration_tests.py --report-only      # Generate reports only
"""

import sys
import os
import subprocess
import json
import yaml
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cluster_integration_tests.log')
    ]
)
logger = logging.getLogger(__name__)


class IntegrationTestConfiguration:
    """Configuration for cluster integration tests"""
    
    def __init__(self):
        # Cluster configuration
        self.cluster_url = os.getenv("CLUSTER_URL", "http://localhost:8000")
        self.mcp_endpoint = os.getenv("MCP_ENDPOINT", f"{self.cluster_url}/mcp")
        self.health_endpoint = os.getenv("HEALTH_ENDPOINT", f"{self.cluster_url}/health")
        
        # Test configuration
        self.test_timeout = int(os.getenv("TEST_TIMEOUT", "600"))  # 10 minutes
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.parallel_workers = int(os.getenv("PARALLEL_WORKERS", "2"))
        
        # Kubernetes configuration
        self.k8s_namespace = os.getenv("K8S_NAMESPACE", "mcp-xgboost")
        self.k8s_context = os.getenv("K8S_CONTEXT", "k3d-mcp-xgboost")
        
        # Report configuration
        self.report_dir = Path(os.getenv("REPORT_DIR", "./test_reports"))
        self.report_dir.mkdir(exist_ok=True)
        
        # Test suite definitions
        self.test_suites = {
            "basic": {
                "name": "Basic Cluster Integration",
                "module": "tests.test_cluster_integration",
                "priority": 1,
                "timeout": 300,
                "description": "Basic connectivity and health checks"
            },
            "agent_workflows": {
                "name": "Agent-Cluster Workflows",
                "module": "tests.test_agent_cluster_workflows",
                "priority": 2,
                "timeout": 600,
                "description": "Agent-to-cluster communication and workflows"
            },
            "end_to_end": {
                "name": "End-to-End ML Workflows",
                "module": "tests.test_end_to_end_ml_workflows",
                "priority": 3,
                "timeout": 900,
                "description": "Complete ML pipelines and production workflows"
            },
            "performance": {
                "name": "Performance and Load Testing",
                "module": "tests.test_cluster_performance_load",
                "priority": 4,
                "timeout": 1200,
                "description": "Performance testing and load scenarios"
            },
            "health": {
                "name": "Cluster Health Verification",
                "module": "tests.test_cluster_health_verification",
                "priority": 5,
                "timeout": 300,
                "description": "Comprehensive cluster health verification"
            }
        }
        
        logger.info(f"Test configuration initialized:")
        logger.info(f"  Cluster URL: {self.cluster_url}")
        logger.info(f"  Test Timeout: {self.test_timeout}s")
        logger.info(f"  Report Directory: {self.report_dir}")


class PrerequisitesChecker:
    """Check prerequisites for running cluster integration tests"""
    
    def __init__(self, config: IntegrationTestConfiguration):
        self.config = config
        self.issues = []
    
    def check_cluster_connectivity(self) -> bool:
        """Check if cluster is reachable"""
        try:
            import requests
            response = requests.get(self.config.health_endpoint, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Cluster connectivity: OK")
                return True
            else:
                self.issues.append(f"Cluster health check failed with status {response.status_code}")
                return False
        except Exception as e:
            self.issues.append(f"Cluster not reachable: {e}")
            return False
    
    def check_python_dependencies(self) -> bool:
        """Check if required Python packages are available"""
        required_packages = [
            "pytest", "requests", "pandas", "numpy", "pyyaml"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.issues.append(f"Missing Python packages: {', '.join(missing_packages)}")
            return False
        
        logger.info("✅ Python dependencies: OK")
        return True
    
    def check_optional_dependencies(self) -> Dict[str, bool]:
        """Check optional dependencies and return availability status"""
        optional_deps = {
            "agent_libraries": ["mcp_agent.core.fastagent"],
            "kubernetes": ["kubernetes"],
            "async_libraries": ["aiohttp", "asyncio"]
        }
        
        availability = {}
        for dep_group, packages in optional_deps.items():
            try:
                for package in packages:
                    __import__(package)
                availability[dep_group] = True
                logger.info(f"✅ {dep_group}: Available")
            except ImportError:
                availability[dep_group] = False
                logger.warning(f"⚠️  {dep_group}: Not available (some tests will be skipped)")
        
        return availability
    
    def check_test_files(self) -> bool:
        """Check if test files exist"""
        test_files = [
            "tests/test_cluster_integration.py",
            "tests/test_agent_cluster_workflows.py",
            "tests/test_end_to_end_ml_workflows.py"
        ]
        
        missing_files = []
        for test_file in test_files:
            if not Path(test_file).exists():
                missing_files.append(test_file)
        
        if missing_files:
            self.issues.append(f"Missing test files: {', '.join(missing_files)}")
            return False
        
        logger.info("✅ Test files: OK")
        return True
    
    def check_kubectl_access(self) -> bool:
        """Check kubectl access to cluster"""
        try:
            result = subprocess.run(
                ["kubectl", "cluster-info", "--context", self.config.k8s_context],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info("✅ kubectl access: OK")
                return True
            else:
                self.issues.append("kubectl access failed")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.issues.append("kubectl not found or timeout")
            return False
    
    def run_all_checks(self) -> bool:
        """Run all prerequisite checks"""
        logger.info("Running prerequisite checks...")
        
        checks = [
            ("Cluster Connectivity", self.check_cluster_connectivity),
            ("Python Dependencies", self.check_python_dependencies),
            ("Test Files", self.check_test_files),
            ("kubectl Access", self.check_kubectl_access)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            logger.info(f"Checking {check_name}...")
            if not check_func():
                all_passed = False
        
        # Check optional dependencies (doesn't affect overall status)
        self.check_optional_dependencies()
        
        if self.issues:
            logger.error("❌ Prerequisite checks failed:")
            for issue in self.issues:
                logger.error(f"  - {issue}")
        else:
            logger.info("✅ All prerequisite checks passed")
        
        return all_passed


class IntegrationTestSuiteRunner:
    """Run individual test suites"""
    
    def __init__(self, config: IntegrationTestConfiguration):
        self.config = config
    
    def run_test_suite(self, suite_name: str, **kwargs) -> Dict[str, Any]:
        """Run a single test suite"""
        suite_config = self.config.test_suites[suite_name]
        
        logger.info(f"Running test suite: {suite_config['name']}")
        logger.info(f"Description: {suite_config['description']}")
        
        start_time = datetime.now()
        
        # Build pytest command
        cmd = [
            "python", "-m", "pytest",
            suite_config["module"].replace(".", "/") + ".py",
            "-v",
            "--tb=short",
            "--color=yes",
            f"--timeout={suite_config['timeout']}",
            "--json-report",
            f"--json-report-file={self.config.report_dir}/{suite_name}_report.json"
        ]
        
        # Add additional arguments
        if kwargs.get("verbose"):
            cmd.append("-vv")
        if kwargs.get("quick"):
            cmd.extend(["-k", "not slow"])
        if kwargs.get("markers"):
            cmd.extend(["-m", kwargs["markers"]])
        
        try:
            # Run the test suite
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=suite_config["timeout"]
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results
            return {
                "suite_name": suite_name,
                "suite_title": suite_config["name"],
                "description": suite_config["description"],
                "status": "passed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "suite_name": suite_name,
                "suite_title": suite_config["name"],
                "description": suite_config["description"],
                "status": "timeout",
                "return_code": -1,
                "duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "stdout": "",
                "stderr": f"Test suite timed out after {suite_config['timeout']} seconds",
                "command": " ".join(cmd)
            }
        
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "suite_name": suite_name,
                "suite_title": suite_config["name"],
                "description": suite_config["description"],
                "status": "error",
                "return_code": -1,
                "duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "stdout": "",
                "stderr": str(e),
                "command": " ".join(cmd)
            }


class IntegrationTestReporter:
    """Generate comprehensive test reports"""
    
    def __init__(self, config: IntegrationTestConfiguration):
        self.config = config
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary report from test results"""
        total_tests = len(results)
        passed_tests = len([r for r in results if r["status"] == "passed"])
        failed_tests = len([r for r in results if r["status"] == "failed"])
        timeout_tests = len([r for r in results if r["status"] == "timeout"])
        error_tests = len([r for r in results if r["status"] == "error"])
        
        total_duration = sum(r["duration"] for r in results)
        
        summary = {
            "test_run_summary": {
                "total_suites": total_tests,
                "passed_suites": passed_tests,
                "failed_suites": failed_tests,
                "timeout_suites": timeout_tests,
                "error_suites": error_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration": total_duration,
                "average_duration": total_duration / total_tests if total_tests > 0 else 0
            },
            "suite_results": results,
            "cluster_info": {
                "cluster_url": self.config.cluster_url,
                "mcp_endpoint": self.config.mcp_endpoint,
                "k8s_namespace": self.config.k8s_namespace,
                "k8s_context": self.config.k8s_context
            },
            "test_configuration": {
                "test_timeout": self.config.test_timeout,
                "max_retries": self.config.max_retries,
                "parallel_workers": self.config.parallel_workers
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return summary
    
    def save_json_report(self, summary: Dict[str, Any]) -> Path:
        """Save JSON report"""
        report_file = self.config.report_dir / f"cluster_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"JSON report saved: {report_file}")
        return report_file
    
    def save_text_report(self, summary: Dict[str, Any]) -> Path:
        """Save human-readable text report"""
        report_file = self.config.report_dir / f"cluster_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("MCP XGBoost Cluster Integration Test Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary section
            summary_data = summary["test_run_summary"]
            f.write(f"Test Run Summary:\n")
            f.write(f"  Total Test Suites: {summary_data['total_suites']}\n")
            f.write(f"  Passed: {summary_data['passed_suites']}\n")
            f.write(f"  Failed: {summary_data['failed_suites']}\n")
            f.write(f"  Timeout: {summary_data['timeout_suites']}\n")
            f.write(f"  Error: {summary_data['error_suites']}\n")
            f.write(f"  Success Rate: {summary_data['success_rate']:.2%}\n")
            f.write(f"  Total Duration: {summary_data['total_duration']:.2f} seconds\n")
            f.write(f"  Average Duration: {summary_data['average_duration']:.2f} seconds\n\n")
            
            # Cluster information
            f.write("Cluster Information:\n")
            cluster_info = summary["cluster_info"]
            for key, value in cluster_info.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            
            # Individual test suite results
            f.write("Test Suite Results:\n")
            f.write("-" * 30 + "\n\n")
            
            for result in summary["suite_results"]:
                status_icon = {
                    "passed": "✅",
                    "failed": "❌",
                    "timeout": "⏰",
                    "error": "💥"
                }.get(result["status"], "❓")
                
                f.write(f"{status_icon} {result['suite_title']}\n")
                f.write(f"   Status: {result['status'].upper()}\n")
                f.write(f"   Duration: {result['duration']:.2f} seconds\n")
                f.write(f"   Description: {result['description']}\n")
                
                if result["status"] != "passed":
                    f.write(f"   Error Output:\n")
                    for line in result["stderr"].split('\n')[:10]:  # First 10 lines
                        f.write(f"     {line}\n")
                    if len(result["stderr"].split('\n')) > 10:
                        f.write("     ... (truncated)\n")
                
                f.write("\n")
            
            f.write(f"Report generated at: {summary['generated_at']}\n")
        
        logger.info(f"Text report saved: {report_file}")
        return report_file
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print summary to console"""
        print("\n" + "=" * 60)
        print("MCP XGBoost Cluster Integration Test Results")
        print("=" * 60)
        
        summary_data = summary["test_run_summary"]
        
        print(f"\nTest Summary:")
        print(f"  Total Suites: {summary_data['total_suites']}")
        print(f"  Passed: {summary_data['passed_suites']} ✅")
        print(f"  Failed: {summary_data['failed_suites']} ❌")
        print(f"  Timeout: {summary_data['timeout_suites']} ⏰")
        print(f"  Error: {summary_data['error_suites']} 💥")
        print(f"  Success Rate: {summary_data['success_rate']:.2%}")
        print(f"  Total Duration: {summary_data['total_duration']:.2f} seconds")
        
        print(f"\nSuite Results:")
        for result in summary["suite_results"]:
            status_icon = {
                "passed": "✅",
                "failed": "❌", 
                "timeout": "⏰",
                "error": "💥"
            }.get(result["status"], "❓")
            
            print(f"  {status_icon} {result['suite_title']} ({result['duration']:.1f}s)")


class ClusterIntegrationTestRunner:
    """Main test runner orchestrator"""
    
    def __init__(self):
        self.config = IntegrationTestConfiguration()
        self.prereq_checker = PrerequisitesChecker(self.config)
        self.suite_runner = IntegrationTestSuiteRunner(self.config)
        self.reporter = IntegrationTestReporter(self.config)
    
    def run_tests(self, suite_names: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Run integration tests"""
        logger.info("Starting MCP XGBoost Cluster Integration Tests")
        logger.info("=" * 60)
        
        # Check prerequisites
        if not kwargs.get("skip_prereqs", False):
            if not self.prereq_checker.run_all_checks():
                logger.error("Prerequisite checks failed. Aborting tests.")
                sys.exit(1)
        
        # Determine which suites to run
        if suite_names is None:
            suite_names = list(self.config.test_suites.keys())
        
        # Sort suites by priority
        suite_names = sorted(
            suite_names,
            key=lambda x: self.config.test_suites[x]["priority"]
        )
        
        logger.info(f"Running test suites: {', '.join(suite_names)}")
        
        # Run test suites
        results = []
        for suite_name in suite_names:
            if suite_name not in self.config.test_suites:
                logger.warning(f"Unknown test suite: {suite_name}")
                continue
            
            result = self.suite_runner.run_test_suite(suite_name, **kwargs)
            results.append(result)
            
            # Log immediate result
            status_icon = {
                "passed": "✅",
                "failed": "❌",
                "timeout": "⏰", 
                "error": "💥"
            }.get(result["status"], "❓")
            
            logger.info(f"{status_icon} {result['suite_title']}: {result['status'].upper()} ({result['duration']:.1f}s)")
            
            # Stop on first failure if requested
            if kwargs.get("fail_fast", False) and result["status"] != "passed":
                logger.warning("Stopping tests due to failure (fail-fast mode)")
                break
        
        # Generate reports
        summary = self.reporter.generate_summary_report(results)
        
        if not kwargs.get("no_reports", False):
            self.reporter.save_json_report(summary)
            self.reporter.save_text_report(summary)
        
        # Print summary
        self.reporter.print_summary(summary)
        
        return summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MCP XGBoost Cluster Integration Test Runner")
    
    parser.add_argument("--suite", choices=["basic", "agent_workflows", "end_to_end", "performance", "health"],
                       help="Run specific test suite")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--skip-prereqs", action="store_true", help="Skip prerequisite checks")
    parser.add_argument("--no-reports", action="store_true", help="Don't generate report files")
    parser.add_argument("--markers", help="Pytest markers to filter tests")
    parser.add_argument("--report-only", action="store_true", help="Generate reports from existing data")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ClusterIntegrationTestRunner()
    
    if args.report_only:
        logger.info("Report-only mode: generating reports from existing test data")
        # This would load existing test results and generate reports
        # Implementation would depend on saved test data format
        return
    
    # Determine suites to run
    suite_names = [args.suite] if args.suite else None
    
    # Run tests
    try:
        summary = runner.run_tests(
            suite_names=suite_names,
            quick=args.quick,
            verbose=args.verbose,
            fail_fast=args.fail_fast,
            skip_prereqs=args.skip_prereqs,
            no_reports=args.no_reports,
            markers=args.markers
        )
        
        # Exit with error code if tests failed
        if summary["test_run_summary"]["failed_suites"] > 0:
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.warning("Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 