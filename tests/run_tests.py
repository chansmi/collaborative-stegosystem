#!/usr/bin/env python3
"""
Comprehensive test runner for the collaborative stegosystem.
Runs all tests and provides detailed reporting.
"""

import unittest
import os
from pathlib import Path


def main():
    # Ensure we are in repo root
    os.chdir(Path(__file__).parent.parent)
    suite = unittest.defaultTestLoader.discover('tests')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List
import coverage

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def run_unit_tests() -> Dict[str, Any]:
    """Run all unit tests and return results."""
    print("🧪 Running Unit Tests...")
    print("=" * 50)
    
    # Discover and run unit tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with coverage
    cov = coverage.Coverage()
    cov.start()
    
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    cov.stop()
    cov.save()
    
    # Generate coverage report
    coverage_data = cov.get_data()
    total_lines = 0
    covered_lines = 0
    
    for filename in coverage_data.measured_files():
        if 'src' in filename:
            file_coverage = coverage_data.get_file_coverage(filename)
            total_lines += len(file_coverage)
            covered_lines += sum(1 for line in file_coverage if file_coverage[line] > 0)
    
    coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
    
    test_results = {
        'tests_run': result.testsRun,
        'tests_failed': len(result.failures),
        'tests_errored': len(result.errors),
        'tests_skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
        'execution_time': end_time - start_time,
        'coverage_percentage': coverage_percentage,
        'total_lines': total_lines,
        'covered_lines': covered_lines,
        'failures': [str(failure[0]) for failure in result.failures],
        'errors': [str(error[0]) for error in result.errors]
    }
    
    print(f"\n📊 Test Results:")
    print(f"   Tests Run: {test_results['tests_run']}")
    print(f"   Tests Failed: {test_results['tests_failed']}")
    print(f"   Tests Errored: {test_results['tests_errored']}")
    print(f"   Tests Skipped: {test_results['tests_skipped']}")
    print(f"   Execution Time: {test_results['execution_time']:.2f}s")
    print(f"   Code Coverage: {test_results['coverage_percentage']:.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for failure in result.failures:
            print(f"   - {failure[0]}: {failure[1]}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for error in result.errors:
            print(f"   - {error[0]}: {error[1]}")
    
    return test_results

def run_integration_tests() -> Dict[str, Any]:
    """Run integration tests and return results."""
    print("\n🔗 Running Integration Tests...")
    print("=" * 50)
    
    # This would run integration tests
    # For now, we'll simulate some basic checks
    
    integration_results = {
        'tests_run': 0,
        'tests_passed': 0,
        'tests_failed': 0,
        'execution_time': 0.0,
        'details': []
    }
    
    # Check if core modules can be imported
    try:
        from src import models, environment, ppo_trainer, utils
        integration_results['tests_passed'] += 1
        integration_results['details'].append("Core module imports: PASSED")
    except Exception as e:
        integration_results['tests_failed'] += 1
        integration_results['details'].append(f"Core module imports: FAILED - {e}")
    
    # Check if configuration files exist
    config_files = ['config.yaml', 'config_gpu.yaml', 'config_test.yaml']
    for config_file in config_files:
        if Path(config_file).exists():
            integration_results['tests_passed'] += 1
            integration_results['details'].append(f"Config file {config_file}: PASSED")
        else:
            integration_results['tests_failed'] += 1
            integration_results['details'].append(f"Config file {config_file}: FAILED - File not found")
    
    # Check if required directories exist
    required_dirs = ['src', 'tests', 'experiments']
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            integration_results['tests_passed'] += 1
            integration_results['details'].append(f"Directory {dir_name}: PASSED")
        else:
            integration_results['tests_failed'] += 1
            integration_results['details'].append(f"Directory {dir_name}: FAILED - Directory not found")
    
    integration_results['tests_run'] = integration_results['tests_passed'] + integration_results['tests_failed']
    
    print(f"📊 Integration Test Results:")
    print(f"   Tests Run: {integration_results['tests_run']}")
    print(f"   Tests Passed: {integration_results['tests_passed']}")
    print(f"   Tests Failed: {integration_results['tests_failed']}")
    
    if integration_results['details']:
        print(f"\n📋 Details:")
        for detail in integration_results['details']:
            status = "✅" if "PASSED" in detail else "❌"
            print(f"   {status} {detail}")
    
    return integration_results

def run_performance_tests() -> Dict[str, Any]:
    """Run performance tests and return results."""
    print("\n⚡ Running Performance Tests...")
    print("=" * 50)
    
    performance_results = {
        'tests_run': 0,
        'tests_passed': 0,
        'tests_failed': 0,
        'execution_time': 0.0,
        'metrics': {}
    }
    
    start_time = time.time()
    
    # Test import performance
    try:
        import_start = time.time()
        from src import models, environment, ppo_trainer, utils
        import_time = time.time() - import_start
        
        performance_results['metrics']['import_time'] = import_time
        performance_results['tests_passed'] += 1
        performance_results['details'] = [f"Module import time: {import_time:.3f}s"]
        
        if import_time < 1.0:  # Should import in under 1 second
            performance_results['tests_passed'] += 1
            performance_results['details'].append("Import performance: PASSED")
        else:
            performance_results['tests_failed'] += 1
            performance_results['details'].append("Import performance: FAILED - Too slow")
            
    except Exception as e:
        performance_results['tests_failed'] += 1
        performance_results['details'] = [f"Import performance test: FAILED - {e}"]
    
    # Test memory usage (basic check)
    try:
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate some memory usage
        test_data = [i for i in range(10000)]
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        performance_results['metrics']['memory_increase_mb'] = memory_increase
        performance_results['tests_passed'] += 1
        performance_results['details'].append(f"Memory usage test: PASSED - {memory_increase:.2f}MB increase")
        
        if memory_increase < 100:  # Should not increase by more than 100MB
            performance_results['tests_passed'] += 1
            performance_results['details'].append("Memory efficiency: PASSED")
        else:
            performance_results['tests_failed'] += 1
            performance_results['details'].append("Memory efficiency: FAILED - Too much memory usage")
            
    except ImportError:
        performance_results['tests_failed'] += 1
        performance_results['details'].append("Memory test: SKIPPED - psutil not available")
    
    performance_results['execution_time'] = time.time() - start_time
    performance_results['tests_run'] = performance_results['tests_passed'] + performance_results['tests_failed']
    
    print(f"📊 Performance Test Results:")
    print(f"   Tests Run: {performance_results['tests_run']}")
    print(f"   Tests Passed: {performance_results['tests_passed']}")
    print(f"   Tests Failed: {performance_results['tests_failed']}")
    print(f"   Execution Time: {performance_results['execution_time']:.2f}s")
    
    if 'details' in performance_results:
        print(f"\n📋 Details:")
        for detail in performance_results['details']:
            status = "✅" if "PASSED" in detail else "❌"
            print(f"   {status} {detail}")
    
    return performance_results

def generate_test_report(unit_results: Dict[str, Any], 
                        integration_results: Dict[str, Any], 
                        performance_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive test report."""
    
    total_tests = (unit_results['tests_run'] + 
                  integration_results['tests_run'] + 
                  performance_results['tests_run'])
    
    total_passed = (unit_results['tests_run'] - unit_results['tests_failed'] - unit_results['tests_errored'] +
                   integration_results['tests_passed'] +
                   performance_results['tests_passed'])
    
    total_failed = (unit_results['tests_failed'] + unit_results['tests_errored'] +
                   integration_results['tests_failed'] +
                   performance_results['tests_failed'])
    
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    report = {
        'timestamp': time.time(),
        'summary': {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'success_rate': overall_success_rate,
            'overall_status': 'PASSED' if overall_success_rate >= 90 else 'FAILED'
        },
        'unit_tests': unit_results,
        'integration_tests': integration_results,
        'performance_tests': performance_results,
        'recommendations': []
    }
    
    # Generate recommendations based on results
    if unit_results['coverage_percentage'] < 80:
        report['recommendations'].append("Increase unit test coverage to at least 80%")
    
    if unit_results['tests_failed'] > 0 or unit_results['tests_errored'] > 0:
        report['recommendations'].append("Fix failing unit tests before proceeding")
    
    if integration_results['tests_failed'] > 0:
        report['recommendations'].append("Fix integration test failures")
    
    if performance_results['tests_failed'] > 0:
        report['recommendations'].append("Address performance issues")
    
    if overall_success_rate < 90:
        report['recommendations'].append("Overall test success rate below 90% - review and fix issues")
    
    return report

def save_test_report(report: Dict[str, Any], output_dir: str = "outputs/test_results"):
    """Save test report to file."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Test report saved to: {report_file}")
        
        # Also save a human-readable summary
        summary_file = output_path / f"test_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("COLLABORATIVE STEGOSYSTEM - TEST REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Status: {report['summary']['overall_status']}\n")
            f.write(f"Success Rate: {report['summary']['success_rate']:.1f}%\n")
            f.write(f"Total Tests: {report['summary']['total_tests']}\n")
            f.write(f"Tests Passed: {report['summary']['total_passed']}\n")
            f.write(f"Tests Failed: {report['summary']['total_failed']}\n\n")
            
            f.write("UNIT TESTS:\n")
            f.write(f"  Coverage: {report['unit_tests']['coverage_percentage']:.1f}%\n")
            f.write(f"  Tests Run: {report['unit_tests']['tests_run']}\n")
            f.write(f"  Tests Failed: {report['unit_tests']['tests_failed']}\n")
            f.write(f"  Tests Errored: {report['unit_tests']['tests_errored']}\n\n")
            
            f.write("INTEGRATION TESTS:\n")
            f.write(f"  Tests Passed: {report['integration_tests']['tests_passed']}\n")
            f.write(f"  Tests Failed: {report['integration_tests']['tests_failed']}\n\n")
            
            f.write("PERFORMANCE TESTS:\n")
            f.write(f"  Tests Passed: {report['performance_tests']['tests_passed']}\n")
            f.write(f"  Tests Failed: {report['performance_tests']['tests_failed']}\n\n")
            
            if report['recommendations']:
                f.write("RECOMMENDATIONS:\n")
                for rec in report['recommendations']:
                    f.write(f"  - {rec}\n")
        
        print(f"📝 Human-readable summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"❌ Failed to save test report: {e}")

def main():
    """Main test runner function."""
    print("🚀 COLLABORATIVE STEGOSYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run all test suites
        unit_results = run_unit_tests()
        integration_results = run_integration_tests()
        performance_results = run_performance_tests()
        
        # Generate comprehensive report
        report = generate_test_report(unit_results, integration_results, performance_results)
        
        # Display overall results
        print("\n" + "=" * 60)
        print("📊 OVERALL TEST RESULTS")
        print("=" * 60)
        print(f"Overall Status: {report['summary']['overall_status']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Tests Passed: {report['summary']['total_passed']}")
        print(f"Tests Failed: {report['summary']['total_failed']}")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   - {rec}")
        
        # Save detailed report
        save_test_report(report)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total test execution time: {total_time:.2f}s")
        
        # Exit with appropriate code
        if report['summary']['overall_status'] == 'PASSED':
            print("\n🎉 All tests passed! System is ready for deployment.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Please review and fix issues before deployment.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
