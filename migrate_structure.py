#!/usr/bin/env python3
"""
Migration script to reorganize the project structure.
This script will help move files from the old structure to the new organized structure.
"""

import shutil
import os
from pathlib import Path
from src.paths import get_project_paths

def migrate_project_structure():
    """Migrate from old structure to new organized structure."""
    paths = get_project_paths()
    
    print("🔄 Starting project structure migration...")
    print(f"Project root: {paths.project_root}")
    print(f"Workspace root: {paths.workspace_root}")
    
    # Create shared directories if they don't exist
    paths.shared_models.mkdir(parents=True, exist_ok=True)
    paths.shared_data.mkdir(parents=True, exist_ok=True)
    
    # Move existing models to shared_models if they're not already there
    old_models_dir = paths.project_root / "models"
    if old_models_dir.exists():
        print(f"📁 Moving models from {old_models_dir} to shared_models...")
        for item in old_models_dir.iterdir():
            if item.is_dir():
                target = paths.shared_models / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
                    print(f"  ✅ Moved {item.name} to shared_models")
                else:
                    print(f"  ⚠️  {item.name} already exists in shared_models, skipping")
    
    # Move existing checkpoints to new structure
    old_checkpoints_dir = paths.project_root / "checkpoints"
    if old_checkpoints_dir.exists():
        print(f"📁 Moving checkpoints from {old_checkpoints_dir} to new structure...")
        for item in old_checkpoints_dir.iterdir():
            if item.is_dir():
                target = paths.checkpoints / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
                    print(f"  ✅ Moved {item.name} to outputs/checkpoints")
                else:
                    print(f"  ⚠️  {item.name} already exists in outputs/checkpoints, skipping")
    
    # Move existing results to new structure
    old_results_dir = paths.project_root / "results"
    if old_results_dir.exists():
        print(f"📁 Moving results from {old_results_dir} to new structure...")
        for item in old_results_dir.iterdir():
            if item.is_dir():
                target = paths.results / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
                    print(f"  ✅ Moved {item.name} to outputs/results")
                else:
                    print(f"  ⚠️  {item.name} already exists in outputs/results, skipping")
    
    # Move existing logs to new structure
    old_logs_dir = paths.project_root / "logs"
    if old_logs_dir.exists():
        print(f"📁 Moving logs from {old_logs_dir} to new structure...")
        for item in old_logs_dir.iterdir():
            if item.is_dir():
                target = paths.logs / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
                    print(f"  ✅ Moved {item.name} to outputs/logs")
                else:
                    print(f"  ⚠️  {item.name} already exists in outputs/logs, skipping")
    
    # Create symbolic links for shared models in the project
    print("🔗 Creating symbolic links for shared models...")
    shared_models_link = paths.project_root / "shared_models"
    if not shared_models_link.exists():
        try:
            shared_models_link.symlink_to(paths.shared_models, target_is_directory=True)
            print(f"  ✅ Created symbolic link: {shared_models_link} -> {paths.shared_models}")
        except OSError as e:
            print(f"  ❌ Failed to create symbolic link: {e}")
    
    # Create symbolic link for shared data
    shared_data_link = paths.project_root / "shared_data"
    if not shared_data_link.exists():
        try:
            shared_data_link.symlink_to(paths.shared_data, target_is_directory=True)
            print(f"  ✅ Created symbolic link: {shared_data_link} -> {paths.shared_data}")
        except OSError as e:
            print(f"  ❌ Failed to create symbolic link: {e}")
    
    print("\n✅ Migration completed!")
    print("\n📋 New structure:")
    print(str(paths))
    
    # Create a summary file
    summary_file = paths.project_root / "MIGRATION_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write("# Project Structure Migration Summary\n\n")
        f.write("## What was moved:\n")
        f.write("- Models → `../shared_models/` (with symbolic link)\n")
        f.write("- Checkpoints → `outputs/checkpoints/`\n")
        f.write("- Results → `outputs/results/`\n")
        f.write("- Logs → `outputs/logs/`\n")
        f.write("- Data → `data/`\n")
        f.write("- Experiments → `experiments/`\n")
        f.write("\n## New structure:\n")
        f.write("```\n")
        f.write(str(paths))
        f.write("\n```\n")
        f.write("\n## Benefits:\n")
        f.write("- Clear separation between shared resources and project outputs\n")
        f.write("- Consistent path structure across all components\n")
        f.write("- Easy to share models between projects\n")
        f.write("- Better organization of project-specific outputs\n")
    
    print(f"\n📄 Migration summary saved to: {summary_file}")

def create_directory_structure():
    """Create the new directory structure."""
    paths = get_project_paths()
    
    print("📁 Creating new directory structure...")
    
    # Create all output directories
    directories = [
        paths.outputs,
        paths.checkpoints,
        paths.loras,
        paths.results,
        paths.logs,
        paths.models,
        paths.experiments,
        paths.data
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    # Create shared directories
    paths.shared_models.mkdir(parents=True, exist_ok=True)
    paths.shared_data.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Created: {paths.shared_models}")
    print(f"  ✅ Created: {paths.shared_data}")
    
    print("✅ Directory structure created!")

if __name__ == "__main__":
    print("🚀 Project Structure Migration Tool")
    print("=" * 50)
    
    # First create the new structure
    create_directory_structure()
    print()
    
    # Then migrate existing files
    migrate_project_structure()
    
    print("\n🎉 All done! Your project is now properly organized.")
    print("\n💡 Next steps:")
    print("1. Update your configuration files to use the new paths")
    print("2. Update your code to use the ProjectPaths class")
    print("3. Test that everything still works correctly")
    print("4. Remove old empty directories if desired")
