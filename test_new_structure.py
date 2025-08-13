#!/usr/bin/env python3
"""
Test script to verify the new project structure and ProjectPaths class.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from paths import get_project_paths

def test_project_paths():
    """Test the ProjectPaths class functionality."""
    print("🧪 Testing ProjectPaths class...")
    
    try:
        paths = get_project_paths()
        print("✅ ProjectPaths instance created successfully")
        
        # Test basic path properties
        print(f"📁 Project Root: {paths.project_root}")
        print(f"📁 Workspace Root: {paths.workspace_root}")
        print(f"📁 Shared Models: {paths.shared_models}")
        print(f"📁 Outputs: {paths.outputs}")
        
        # Test path creation methods
        test_checkpoint = paths.get_checkpoint_path("test_checkpoint")
        test_lora = paths.get_lora_path("test_lora")
        test_model = paths.get_model_path("test_model")
        test_log = paths.get_log_path("test.log")
        test_result = paths.get_result_path("test_result")
        
        print(f"📁 Test Checkpoint Path: {test_checkpoint}")
        print(f"📁 Test LoRA Path: {test_lora}")
        print(f"📁 Test Model Path: {test_model}")
        print(f"📁 Test Log Path: {test_log}")
        print(f"📁 Test Result Path: {test_result}")
        
        # Test that directories exist
        print("\n🔍 Checking directory existence...")
        directories_to_check = [
            paths.outputs,
            paths.checkpoints,
            paths.loras,
            paths.results,
            paths.logs,
            paths.models,
            paths.experiments,
            paths.data,
            paths.shared_models,
            paths.shared_data
        ]
        
        for directory in directories_to_check:
            if directory.exists():
                print(f"  ✅ {directory.name}: {directory}")
            else:
                print(f"  ❌ {directory.name}: {directory} (MISSING)")
        
        # Test symbolic links
        print("\n🔗 Checking symbolic links...")
        shared_models_link = paths.project_root / "shared_models"
        shared_data_link = paths.project_root / "shared_data"
        
        if shared_models_link.is_symlink():
            print(f"  ✅ shared_models link: {shared_models_link} -> {shared_models_link.resolve()}")
        else:
            print(f"  ❌ shared_models link not found")
            
        if shared_data_link.is_symlink():
            print(f"  ✅ shared_data link: {shared_data_link} -> {shared_data_link.resolve()}")
        else:
            print(f"  ❌ shared_data link not found")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_path_operations():
    """Test path operations and file creation."""
    print("\n🧪 Testing path operations...")
    
    try:
        paths = get_project_paths()
        
        # Test creating a test file
        test_file = paths.logs / "test_structure.log"
        test_content = "This is a test file to verify the new structure works."
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        print(f"✅ Created test file: {test_file}")
        
        # Verify file was created
        if test_file.exists():
            print(f"✅ Test file exists and is readable")
            
            # Read back content
            with open(test_file, 'r') as f:
                content = f.read()
            
            if content == test_content:
                print("✅ File content matches expected content")
            else:
                print("❌ File content mismatch")
        else:
            print("❌ Test file was not created")
        
        # Clean up test file
        test_file.unlink()
        print("✅ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Path operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 Testing New Project Structure")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_project_paths()
    test2_passed = test_path_operations()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"ProjectPaths Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Path Operations Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The new structure is working correctly.")
        print("\n💡 Next steps:")
        print("1. Update your training scripts to use the new paths")
        print("2. Test your main training pipeline")
        print("3. Remove old empty directories if desired")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
