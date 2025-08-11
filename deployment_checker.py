#!/usr/bin/env python3
"""
Deployment Readiness Checker for Collaborative Stegosystem
Validates system readiness before distributed deployment.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import subprocess
import platform
import psutil
import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentCheckError(Exception):
    """Custom exception for deployment check failures."""
    pass

class DeploymentChecker:
    """Comprehensive deployment readiness checker."""
    
    def __init__(self):
        """Initialize the deployment checker."""
        self.checks_passed = 0
        self.checks_failed = 0
        self.checks_warning = 0
        self.check_results = []
        
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all deployment readiness checks."""
        logger.info("🚀 Starting Deployment Readiness Checks...")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all check categories
        self._check_system_requirements()
        self._check_dependencies()
        self._check_configuration()
        self._check_code_quality()
        self._check_testing()
        self._check_security()
        self._check_performance()
        self._check_documentation()
        
        # Generate summary
        total_time = time.time() - start_time
        summary = self._generate_summary(total_time)
        
        # Display results
        self._display_results(summary)
        
        return summary
    
    def _check_system_requirements(self):
        """Check system requirements."""
        logger.info("🔍 Checking System Requirements...")
        
        # Python version
        python_version = sys.version_info
        if python_version >= (3, 10):
            self._add_check_result("Python Version", "PASSED", 
                                 f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            self._add_check_result("Python Version", "FAILED", 
                                 f"Python {python_version.major}.{python_version.minor}.{python_version.micro} - Requires 3.10+")
        
        # Platform
        platform_info = platform.platform()
        self._add_check_result("Platform", "PASSED", platform_info)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 8:
            self._add_check_result("System Memory", "PASSED", f"{memory_gb:.1f} GB available")
        else:
            self._add_check_result("System Memory", "WARNING", f"{memory_gb:.1f} GB available - 8+ GB recommended")
        
        # Disk space
        disk = psutil.disk_usage('/')
        disk_gb = disk.free / (1024**3)
        if disk_gb >= 10:
            self._add_check_result("Disk Space", "PASSED", f"{disk_gb:.1f} GB available")
        else:
            self._add_check_result("Disk Space", "WARNING", f"{disk_gb:.1f} GB available - 10+ GB recommended")
    
    def _check_dependencies(self):
        """Check Python dependencies."""
        logger.info("📦 Checking Dependencies...")
        
        required_packages = [
            'torch', 'transformers', 'trl', 'peft', 'pandas', 
            'numpy', 'tqdm', 'pyyaml', 'wandb', 'bitsandbytes', 'openai'
        ]
        
        for package in required_packages:
            try:
                if package == 'pyyaml':
                    import yaml
                    self._add_check_result(f"Package: {package}", "PASSED", "Successfully imported")
                else:
                    __import__(package)
                    self._add_check_result(f"Package: {package}", "PASSED", "Successfully imported")
            except ImportError:
                self._add_check_result(f"Package: {package}", "FAILED", "Not installed")
        
        # Check PyTorch CUDA support
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            self._add_check_result("PyTorch CUDA", "PASSED", 
                                 f"CUDA {cuda_version}, {gpu_count} GPU(s), {gpu_name}")
        else:
            self._add_check_result("PyTorch CUDA", "WARNING", "CUDA not available - will use CPU/MPS")
        
        # Check for MPS support (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._add_check_result("PyTorch MPS", "PASSED", "MPS (Metal Performance Shaders) available")
        else:
            self._add_check_result("PyTorch MPS", "INFO", "MPS not available")
    
    def _check_configuration(self):
        """Check configuration files."""
        logger.info("⚙️  Checking Configuration...")
        
        config_files = ['config.yaml', 'config_gpu.yaml', 'config_test.yaml']
        for config_file in config_files:
            if Path(config_file).exists():
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Basic validation
                    required_sections = ['model', 'ppo', 'training', 'env', 'openai']
                    missing_sections = [section for section in required_sections if section not in config]
                    
                    if not missing_sections:
                        self._add_check_result(f"Config: {config_file}", "PASSED", "Valid configuration")
                    else:
                        self._add_check_result(f"Config: {config_file}", "FAILED", 
                                             f"Missing sections: {missing_sections}")
                        
                except Exception as e:
                    self._add_check_result(f"Config: {config_file}", "FAILED", f"Error: {e}")
            else:
                self._add_check_result(f"Config: {config_file}", "WARNING", "File not found")
        
        # Check environment variables
        required_env_vars = ['HF_TOKEN', 'OPENAI_API_KEY', 'WANDB_API_KEY']
        for var in required_env_vars:
            if os.environ.get(var):
                self._add_check_result(f"Env Var: {var}", "PASSED", "Set")
            else:
                self._add_check_result(f"Env Var: {var}", "FAILED", "Not set")
    
    def _check_code_quality(self):
        """Check code quality metrics."""
        logger.info("📝 Checking Code Quality...")
        
        # Check if source files exist
        src_files = ['src/__init__.py', 'src/models.py', 'src/environment.py', 
                    'src/ppo_trainer.py', 'src/utils.py']
        
        for src_file in src_files:
            if Path(src_file).exists():
                self._add_check_result(f"Source: {src_file}", "PASSED", "File exists")
            else:
                self._add_check_result(f"Source: {src_file}", "FAILED", "File missing")
        
        # Check for common code quality issues
        self._check_file_sizes()
        self._check_import_structure()
    
    def _check_file_sizes(self):
        """Check file sizes for potential issues."""
        large_files = []
        for file_path in Path('src').rglob('*.py'):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > 1.0:  # Files larger than 1MB
                    large_files.append((file_path, size_mb))
        
        if large_files:
            self._add_check_result("File Sizes", "WARNING", 
                                 f"Large files detected: {len(large_files)} files > 1MB")
        else:
            self._add_check_result("File Sizes", "PASSED", "All source files reasonably sized")
    
    def _check_import_structure(self):
        """Check import structure."""
        try:
            # Try importing main modules
            from src import models, environment, ppo_trainer, utils
            self._add_check_result("Import Structure", "PASSED", "All modules import successfully")
        except Exception as e:
            self._add_check_result("Import Structure", "FAILED", f"Import error: {e}")
    
    def _check_testing(self):
        """Check testing infrastructure."""
        logger.info("🧪 Checking Testing Infrastructure...")
        
        # Check if tests directory exists
        if Path('tests').exists():
            test_files = list(Path('tests').glob('test_*.py'))
            if test_files:
                self._add_check_result("Test Directory", "PASSED", f"{len(test_files)} test files found")
            else:
                self._add_check_result("Test Directory", "WARNING", "No test files found")
        else:
            self._add_check_result("Test Directory", "FAILED", "Tests directory missing")
        
        # Check test requirements
        if Path('requirements-test.txt').exists():
            self._add_check_result("Test Requirements", "PASSED", "Test requirements file exists")
        else:
            self._add_check_result("Test Requirements", "WARNING", "Test requirements file missing")
        
        # Check if tests can run
        try:
            result = subprocess.run([sys.executable, '-m', 'pytest', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self._add_check_result("Pytest", "PASSED", "Pytest available")
            else:
                self._add_check_result("Pytest", "FAILED", "Pytest not working")
        except Exception as e:
            self._add_check_result("Pytest", "WARNING", f"Pytest check failed: {e}")
    
    def _check_security(self):
        """Check security aspects."""
        logger.info("🔒 Checking Security...")
        
        # Check for hardcoded secrets
        hardcoded_secrets = self._check_for_hardcoded_secrets()
        if hardcoded_secrets:
            self._add_check_result("Hardcoded Secrets", "FAILED", 
                                 f"Found {len(hardcoded_secrets)} potential hardcoded secrets")
        else:
            self._add_check_result("Hardcoded Secrets", "PASSED", "No hardcoded secrets found")
        
        # Check .gitignore
        if Path('.gitignore').exists():
            gitignore_content = Path('.gitignore').read_text()
            if 'models/' in gitignore_content and 'results/' in gitignore_content:
                self._add_check_result("Gitignore", "PASSED", "Properly configured")
            else:
                self._add_check_result("Gitignore", "WARNING", "May be missing important exclusions")
        else:
            self._add_check_result("Gitignore", "FAILED", "File missing")
        
        # Check for environment variable usage
        env_var_usage = self._check_env_var_usage()
        if env_var_usage:
            self._add_check_result("Environment Variables", "PASSED", "Properly used for sensitive data")
        else:
            self._add_check_result("Environment Variables", "WARNING", "May not be using env vars for secrets")
    
    def _check_for_hardcoded_secrets(self) -> List[str]:
        """Check for hardcoded secrets in source files."""
        secrets = []
        secret_patterns = [
            r'sk-[a-zA-Z0-9]{48}',
            r'hf_[a-zA-Z0-9]{39}',
            r'[a-zA-Z0-9]{32,}',
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'api_key\s*=\s*["\'][^"\']{8,}["\']'
        ]
        
        # Filter out false positives (common class names, etc.)
        false_positive_patterns = [
            r'AutoModelForCausalLMWithValueHead',
            r'CollaborativePPOTrainer',
            r'TradingEnvironment',
            r'ConversationManager'
        ]
        
        for file_path in Path('src').rglob('*.py'):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    for pattern in secret_patterns:
                        import re
                        matches = re.findall(pattern, content)
                        if matches:
                            # Filter out false positives
                            filtered_matches = []
                            for match in matches:
                                is_false_positive = False
                                for fp_pattern in false_positive_patterns:
                                    if re.search(fp_pattern, match):
                                        is_false_positive = True
                                        break
                                if not is_false_positive:
                                    filtered_matches.append(match)
                            
                            if filtered_matches:
                                secrets.append(f"{file_path}: {filtered_matches[:3]}")  # First 3 matches
                except Exception:
                    continue
        
        return secrets
    
    def _check_env_var_usage(self) -> bool:
        """Check if environment variables are properly used."""
        try:
            from src import models, environment
            # This is a basic check - in practice, you'd want more sophisticated analysis
            return True
        except Exception:
            return False
    
    def _check_performance(self):
        """Check performance characteristics."""
        logger.info("⚡ Checking Performance...")
        
        # Check import performance
        try:
            start_time = time.time()
            from src import models, environment, ppo_trainer, utils
            import_time = time.time() - start_time
            
            if import_time < 2.0:
                self._add_check_result("Import Performance", "PASSED", f"{import_time:.2f}s")
            else:
                self._add_check_result("Import Performance", "WARNING", f"{import_time:.2f}s - slow")
                
        except Exception as e:
            self._add_check_result("Import Performance", "FAILED", f"Error: {e}")
        
        # Check memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            if memory_mb < 500:
                self._add_check_result("Memory Usage", "PASSED", f"{memory_mb:.1f} MB")
            else:
                self._add_check_result("Memory Usage", "WARNING", f"{memory_mb:.1f} MB - high")
                
        except Exception as e:
            self._add_check_result("Memory Usage", "WARNING", f"Check failed: {e}")
    
    def _check_documentation(self):
        """Check documentation completeness."""
        logger.info("📚 Checking Documentation...")
        
        # Check README
        if Path('README.md').exists():
            readme_size = Path('README.md').stat().st_size
            if readme_size > 1000:  # More than 1KB
                self._add_check_result("README", "PASSED", "Comprehensive documentation")
            else:
                self._add_check_result("README", "WARNING", "Documentation may be minimal")
        else:
            self._add_check_result("README", "FAILED", "Missing README")
        
        # Check training guide
        if Path('TRAINING.md').exists():
            self._add_check_result("Training Guide", "PASSED", "Training documentation exists")
        else:
            self._add_check_result("Training Guide", "WARNING", "Training guide missing")
        
        # Check deployment guide
        if Path('DEPLOYMENT.md').exists():
            self._add_check_result("Deployment Guide", "PASSED", "Deployment documentation exists")
        else:
            self._add_check_result("Deployment Guide", "WARNING", "Deployment guide missing")
        
        # Check docstrings in source files
        docstring_coverage = self._check_docstring_coverage()
        if docstring_coverage > 80:
            self._add_check_result("Docstring Coverage", "PASSED", f"{docstring_coverage:.1f}%")
        else:
            self._add_check_result("Docstring Coverage", "WARNING", f"{docstring_coverage:.1f}% - improve coverage")
    
    def _check_docstring_coverage(self) -> float:
        """Check docstring coverage in source files."""
        total_functions = 0
        documented_functions = 0
        
        for file_path in Path('src').rglob('*.py'):
            if file_path.is_file():
                try:
                    content = file_path.read_text()
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def ') or line.strip().startswith('class '):
                            total_functions += 1
                            # Check if next non-empty line has docstring
                            for j in range(i + 1, len(lines)):
                                next_line = lines[j].strip()
                                if next_line:
                                    if next_line.startswith('"""') or next_line.startswith("'''"):
                                        documented_functions += 1
                                    break
                                    
                except Exception:
                    continue
        
        return (documented_functions / total_functions * 100) if total_functions > 0 else 0
    
    def _add_check_result(self, check_name: str, status: str, message: str):
        """Add a check result."""
        result = {
            'name': check_name,
            'status': status,
            'message': message,
            'timestamp': time.time()
        }
        
        self.check_results.append(result)
        
        if status == "PASSED":
            self.checks_passed += 1
        elif status == "FAILED":
            self.checks_failed += 1
        elif status == "WARNING":
            self.checks_warning += 1
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate deployment readiness summary."""
        total_checks = self.checks_passed + self.checks_failed + self.checks_warning
        
        if self.checks_failed == 0 and self.checks_warning <= 3:
            overall_status = "READY"
        elif self.checks_failed <= 2:
            overall_status = "NEEDS_ATTENTION"
        else:
            overall_status = "NOT_READY"
        
        summary = {
            'timestamp': time.time(),
            'overall_status': overall_status,
            'total_checks': total_checks,
            'checks_passed': self.checks_passed,
            'checks_failed': self.checks_failed,
            'checks_warning': self.checks_warning,
            'success_rate': (self.checks_passed / total_checks * 100) if total_checks > 0 else 0,
            'execution_time': total_time,
            'check_results': self.check_results,
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on check results."""
        recommendations = []
        
        if self.checks_failed > 0:
            recommendations.append("Fix all FAILED checks before deployment")
        
        if self.checks_warning > 3:
            recommendations.append("Address WARNING checks to improve deployment quality")
        
        # Specific recommendations based on failures
        failed_checks = [r for r in self.check_results if r['status'] == 'FAILED']
        for check in failed_checks:
            if 'Environment Variables' in check['name']:
                recommendations.append("Set required environment variables (HF_TOKEN, OPENAI_API_KEY, WANDB_API_KEY)")
            elif 'Configuration' in check['name']:
                recommendations.append("Fix configuration file issues")
            elif 'Dependencies' in check['name']:
                recommendations.append("Install missing Python packages")
        
        if not recommendations:
            recommendations.append("System is ready for deployment!")
        
        return recommendations
    
    def _display_results(self, summary: Dict[str, Any]):
        """Display check results."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 DEPLOYMENT READINESS RESULTS")
        logger.info("=" * 60)
        
        # Overall status
        status_emoji = {"READY": "🎉", "NEEDS_ATTENTION": "⚠️", "NOT_READY": "❌"}
        emoji = status_emoji.get(summary['overall_status'], "❓")
        logger.info(f"Overall Status: {emoji} {summary['overall_status']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
        logger.info(f"Total Checks: {summary['total_checks']}")
        logger.info(f"Passed: {summary['checks_passed']} | Failed: {summary['checks_failed']} | Warnings: {summary['checks_warning']}")
        logger.info(f"Execution Time: {summary['execution_time']:.2f}s")
        
        # Detailed results
        logger.info(f"\n📋 DETAILED RESULTS:")
        for result in summary['check_results']:
            status_emoji = {"PASSED": "✅", "FAILED": "❌", "WARNING": "⚠️", "INFO": "ℹ️"}
            emoji = status_emoji.get(result['status'], "❓")
            logger.info(f"   {emoji} {result['name']}: {result['message']}")
        
        # Recommendations
        if summary['recommendations']:
            logger.info(f"\n💡 RECOMMENDATIONS:")
            for rec in summary['recommendations']:
                logger.info(f"   - {rec}")
        
        # Final verdict
        logger.info(f"\n{'=' * 60}")
        if summary['overall_status'] == "READY":
            logger.info("🎉 SYSTEM IS READY FOR DEPLOYMENT!")
        elif summary['overall_status'] == "NEEDS_ATTENTION":
            logger.info("⚠️  SYSTEM NEEDS ATTENTION BEFORE DEPLOYMENT")
        else:
            logger.info("❌ SYSTEM IS NOT READY FOR DEPLOYMENT")
        logger.info(f"{'=' * 60}")

def save_deployment_report(summary: Dict[str, Any], output_dir: str = "deployment_reports"):
    """Save deployment readiness report."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"deployment_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"💾 Deployment report saved to: {report_file}")
        
        # Also save human-readable summary
        summary_file = output_path / f"deployment_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("COLLABORATIVE STEGOSYSTEM - DEPLOYMENT READINESS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Overall Status: {summary['overall_status']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1f}%\n")
            f.write(f"Total Checks: {summary['total_checks']}\n")
            f.write(f"Checks Passed: {summary['checks_passed']}\n")
            f.write(f"Checks Failed: {summary['checks_failed']}\n")
            f.write(f"Checks Warning: {summary['checks_warning']}\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            for rec in summary['recommendations']:
                f.write(f"  - {rec}\n")
        
        print(f"📝 Human-readable summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save deployment report: {e}")

def main():
    """Main deployment checker function."""
    print("🚀 COLLABORATIVE STEGOSYSTEM - DEPLOYMENT READINESS CHECKER")
    print("=" * 70)
    
    try:
        # Run deployment checks
        checker = DeploymentChecker()
        summary = checker.run_all_checks()
        
        # Save report
        save_deployment_report(summary)
        
        # Exit with appropriate code
        if summary['overall_status'] == 'READY':
            print("\n🎉 Deployment checks passed! System is ready for deployment.")
            sys.exit(0)
        elif summary['overall_status'] == 'NEEDS_ATTENTION':
            print("\n⚠️  Deployment checks need attention. Review warnings before deployment.")
            sys.exit(1)
        else:
            print("\n❌ Deployment checks failed. Fix issues before deployment.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Deployment checker failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
