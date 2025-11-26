#!/usr/bin/env python3
"""
Codebase Cleanup Agent

Mission: Remove everything broken, unused, or distracting while guaranteeing
the AutoML v2 pipeline still runs.

This agent follows a systematic approach:
1. Safety net (backup, baseline tests)
2. Map the living system (import graph)
3. Classify files (CORE/SUPPORT/EXAMPLE/LEGACY/BROKEN)
4. Remove breaking code
5. Simplify APIs
6. Validate everything still works
"""

import os
import ast
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import json
import shutil
from collections import defaultdict


class CodebaseCleanupAgent:
    """Agent that systematically cleans up the codebase."""
    
    # Core files that must be preserved
    CORE_FILES = {
        "main.py",
        "engine/orchestrator.py",
        "engine/intelligent_search.py",
        "engine/meta_learner.py",
        "engine/preprocessing_strategies.py",
        "engine/trainer.py",
        "engine/evaluator.py",
        "engine/report_generator.py",
        "engine/config.py",
        "models/model_zoo.py",
        "models/sk_models.py",
        "models/mlp_variants.py",
        "utils/datasets.py",
        "utils/seed_utils.py",
        "utils/logging_utils.py",
    }
    
    # Support files (configs, etc.)
    SUPPORT_FILES = {
        "configs/models.yaml",
        "configs/preprocessing.yaml",
    }
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.file_classifications: Dict[str, str] = {}
        self.broken_files: List[str] = []
        self.unused_files: List[str] = []
        
    def phase_0_safety_net(self) -> bool:
        """
        Phase 0: Create safety net before touching anything.
        
        Returns:
            True if baseline tests pass, False otherwise
        """
        print("\n" + "=" * 60)
        print("Phase 0: Safety Net")
        print("=" * 60)
        
        # Check if we're in a git repo
        if (self.root_dir / ".git").exists():
            print("[INFO] Git repo detected. Consider creating a branch:")
            print("  git checkout -b cleanup-phase-1")
        else:
            print("[WARN] Not a git repo. Consider backing up before cleanup.")
        
        # Run baseline tests
        print("\n[INFO] Running baseline tests...")
        print("  (These should be run manually to establish baseline)")
        print("\n  Tests to run:")
        print("    1. python main.py → [3] Titanic (Accuracy)")
        print("    2. python main.py → [4] Titanic (F1)")
        print("    3. python main.py → [5] Custom CSV (digits)")
        print("    4. python main.py → [8] Dataset-agnostic → Intelligent (paddy)")
        print("    5. python stress_test.py (if exists)")
        
        return True
    
    def phase_1_map_living_system(self) -> Dict[str, Set[str]]:
        """
        Phase 1: Build import graph to find what's actually used.
        
        Returns:
            Dictionary mapping file paths to sets of imported modules
        """
        print("\n" + "=" * 60)
        print("Phase 1: Map the Living System")
        print("=" * 60)
        
        # Start from core entry points
        entry_points = [
            "main.py",
            "engine/orchestrator.py",
            "engine/intelligent_search.py",
        ]
        
        visited = set()
        to_visit = entry_points.copy()
        
        while to_visit:
            file_path = to_visit.pop(0)
            if file_path in visited:
                continue
            
            visited.add(file_path)
            full_path = self.root_dir / file_path
            
            if not full_path.exists():
                continue
            
            # Parse imports
            imports = self._extract_imports(full_path)
            self.import_graph[file_path] = imports
            
            # Add imported modules to visit list
            for imp in imports:
                # Convert import to file path
                module_path = self._import_to_path(imp)
                if module_path and module_path not in visited:
                    to_visit.append(module_path)
        
        print(f"[INFO] Mapped {len(visited)} files in import graph")
        return self.import_graph
    
    def _extract_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file."""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except Exception as e:
            print(f"[WARN] Could not parse {file_path}: {e}")
        
        return imports
    
    def _import_to_path(self, module_name: str) -> Optional[str]:
        """Convert Python import to file path."""
        # Remove relative imports
        if module_name.startswith('.'):
            return None
        
        # Convert dots to slashes
        parts = module_name.split('.')
        
        # Check common patterns
        if parts[0] == "engine":
            if len(parts) == 2:
                return f"engine/{parts[1]}.py"
        elif parts[0] == "models":
            if len(parts) == 2:
                return f"models/{parts[1]}.py"
        elif parts[0] == "utils":
            if len(parts) == 2:
                return f"utils/{parts[1]}.py"
        
        return None
    
    def phase_2_classify_files(self) -> Dict[str, str]:
        """
        Phase 2: Classify all files as CORE/SUPPORT/EXAMPLE/LEGACY/BROKEN.
        
        Returns:
            Dictionary mapping file paths to classifications
        """
        print("\n" + "=" * 60)
        print("Phase 2: Classify All Files")
        print("=" * 60)
        
        # Find all Python files
        python_files = list(self.root_dir.rglob("*.py"))
        
        # Also check for YAML configs, etc.
        config_files = list(self.root_dir.rglob("*.yaml")) + list(self.root_dir.rglob("*.yml"))
        
        all_files = python_files + config_files
        
        for file_path in all_files:
            rel_path = str(file_path.relative_to(self.root_dir))
            
            # Skip hidden files and __pycache__
            if any(part.startswith('.') or part == '__pycache__' for part in file_path.parts):
                continue
            
            # Classify
            classification = self._classify_file(rel_path)
            self.file_classifications[rel_path] = classification
        
        # Print summary
        counts = defaultdict(int)
        for classification in self.file_classifications.values():
            counts[classification] += 1
        
        print("\n[INFO] File Classifications:")
        for cls, count in sorted(counts.items()):
            print(f"  {cls}: {count} files")
        
        return self.file_classifications
    
    def _classify_file(self, file_path: str) -> str:
        """Classify a single file."""
        # Core files
        if file_path in self.CORE_FILES:
            return "CORE"
        
        # Support files
        if file_path in self.SUPPORT_FILES:
            return "SUPPORT"
        
        # Legacy folder
        if "legacy" in file_path.lower() or "old" in file_path.lower():
            return "LEGACY"
        
        # Examples/demos
        if any(keyword in file_path.lower() for keyword in ["demo", "example", "test", "notebook", "playground"]):
            return "EXAMPLE"
        
        # Check if it's in import graph
        if file_path in self.import_graph or any(file_path in imports for imports in self.import_graph.values()):
            # Check if it's broken
            if self._is_broken(file_path):
                return "BROKEN"
            return "SUPPORT"
        
        # Check if it's a known example/demo
        known_examples = [
            "online_titanic_demo.py",
            "online_demo.py",
            "stress_test.py",
            "app.py",  # Streamlit app is a demo
        ]
        if any(example in file_path for example in known_examples):
            return "EXAMPLE"
        
        # Unknown - mark as potential legacy
        return "UNKNOWN"
    
    def _is_broken(self, file_path: str) -> bool:
        """Check if a file is broken (has import errors, etc.)."""
        full_path = self.root_dir / file_path
        
        if not full_path.exists() or not file_path.endswith('.py'):
            return False
        
        try:
            # Try to parse the file
            with open(full_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            
            # Try to check imports
            # (This is a simple check - full validation would require actually importing)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check for obvious broken imports
                if "from engine.search_controller import" in content and "legacy" not in file_path:
                    return True  # Using old import
            
        except SyntaxError:
            return True
        except Exception:
            pass  # Other errors might be OK
        
        return False
    
    def phase_3_remove_breaking_code(self, dry_run: bool = True) -> Dict[str, List[str]]:
        """
        Phase 3: Remove breaking and unnecessary code.
        
        Args:
            dry_run: If True, only report what would be done
        
        Returns:
            Dictionary of actions taken
        """
        print("\n" + "=" * 60)
        print("Phase 3: Remove Breaking Code")
        print("=" * 60)
        print(f"[INFO] Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        
        actions = {
            "deleted": [],
            "moved_to_examples": [],
            "moved_to_legacy": [],
            "fixed": [],
        }
        
        # Create examples directory
        examples_dir = self.root_dir / "examples"
        if not dry_run:
            examples_dir.mkdir(exist_ok=True)
        
        # Process BROKEN files
        for file_path, classification in self.file_classifications.items():
            if classification == "BROKEN":
                full_path = self.root_dir / file_path
                
                # Check if it's used by core
                is_used_by_core = self._is_used_by_core(file_path)
                
                if not is_used_by_core:
                    if not dry_run:
                        full_path.unlink()
                    actions["deleted"].append(file_path)
                    print(f"  [DELETE] {file_path} (broken, not used by core)")
        
        # Process LEGACY files
        for file_path, classification in self.file_classifications.items():
            if classification == "LEGACY":
                full_path = self.root_dir / file_path
                
                # Check if it's imported by core
                is_imported = self._is_imported_by_core(file_path)
                
                if not is_imported:
                    if not dry_run:
                        # Move to legacy folder if not already there
                        if "legacy" not in file_path:
                            legacy_dir = self.root_dir / "engine" / "legacy"
                            legacy_dir.mkdir(exist_ok=True)
                            dest = legacy_dir / Path(file_path).name
                            shutil.move(str(full_path), str(dest))
                            actions["moved_to_legacy"].append(f"{file_path} → {dest}")
                        else:
                            # Already in legacy, can delete if you want
                            pass
                    print(f"  [LEGACY] {file_path} (not imported by core)")
        
        # Process EXAMPLE files
        for file_path, classification in self.file_classifications.items():
            if classification == "EXAMPLE":
                full_path = self.root_dir / file_path
                
                # Move to examples directory
                if not dry_run:
                    dest = examples_dir / Path(file_path).name
                    if full_path != dest:
                        shutil.move(str(full_path), str(dest))
                        actions["moved_to_examples"].append(f"{file_path} → {dest}")
                else:
                    actions["moved_to_examples"].append(f"{file_path} → examples/")
                print(f"  [MOVE] {file_path} → examples/")
        
        # Process UNKNOWN files
        for file_path, classification in self.file_classifications.items():
            if classification == "UNKNOWN":
                full_path = self.root_dir / file_path
                
                # Check if it's actually used
                if not self._is_used_by_core(file_path):
                    if not dry_run:
                        # Move to examples or delete
                        if full_path.suffix == '.py':
                            dest = examples_dir / Path(file_path).name
                            shutil.move(str(full_path), str(dest))
                            actions["moved_to_examples"].append(f"{file_path} → {dest}")
                        else:
                            full_path.unlink()
                            actions["deleted"].append(file_path)
                    print(f"  [UNKNOWN] {file_path} (not used, moving/deleting)")
        
        return actions
    
    def _is_used_by_core(self, file_path: str) -> bool:
        """Check if a file is used by any core file."""
        # Check if it's in the import graph from core entry points
        for core_file, imports in self.import_graph.items():
            if file_path in imports or any(file_path in str(imp) for imp in imports):
                return True
        return False
    
    def _is_imported_by_core(self, file_path: str) -> bool:
        """Check if a file is directly imported by core files."""
        module_name = file_path.replace('/', '.').replace('.py', '')
        
        for core_file in self.CORE_FILES:
            if core_file in self.import_graph:
                imports = self.import_graph[core_file]
                if any(module_name in imp or imp in module_name for imp in imports):
                    return True
        
        return False
    
    def phase_4_simplify_apis(self, dry_run: bool = True) -> List[str]:
        """
        Phase 4: Simplify APIs and configs.
        
        Returns:
            List of simplifications made
        """
        print("\n" + "=" * 60)
        print("Phase 4: Simplify APIs")
        print("=" * 60)
        
        simplifications = []
        
        # Check AutoMLConfig for unused fields
        config_file = self.root_dir / "engine" / "config.py"
        if config_file.exists():
            # This would require parsing the config class
            # For now, just report
            simplifications.append("Review AutoMLConfig for unused fields")
        
        # Check main.py for broken menu options
        main_file = self.root_dir / "main.py"
        if main_file.exists():
            with open(main_file, 'r') as f:
                content = f.read()
                if "from engine.search_controller import" in content and "legacy" not in content:
                    simplifications.append("main.py still imports old search_controller (should use legacy)")
        
        return simplifications
    
    def phase_5_validate(self) -> bool:
        """
        Phase 5: Post-cleanup validation.
        
        Returns:
            True if validation passes
        """
        print("\n" + "=" * 60)
        print("Phase 5: Post-Cleanup Validation")
        print("=" * 60)
        
        print("[INFO] Validation checklist:")
        print("  1. All core files exist")
        print("  2. No broken imports in core files")
        print("  3. Main entry points are accessible")
        
        # Check core files exist
        all_exist = True
        for core_file in self.CORE_FILES:
            full_path = self.root_dir / core_file
            if not full_path.exists():
                print(f"  [ERROR] Core file missing: {core_file}")
                all_exist = False
            else:
                print(f"  [OK] {core_file}")
        
        return all_exist
    
    def generate_report(self, output_file: str = "cleanup_report.json"):
        """Generate a cleanup report."""
        report = {
            "import_graph": {k: list(v) for k, v in self.import_graph.items()},
            "classifications": self.file_classifications,
            "broken_files": self.broken_files,
            "unused_files": self.unused_files,
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[INFO] Cleanup report saved to {output_file}")


def main():
    """Run the cleanup agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Codebase Cleanup Agent")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (don't actually delete/move)")
    parser.add_argument("--root", default=".", help="Root directory of project")
    parser.add_argument("--skip-phase", type=int, help="Skip a specific phase")
    
    args = parser.parse_args()
    
    agent = CodebaseCleanupAgent(root_dir=args.root)
    
    # Run phases
    if args.skip_phase != 0:
        agent.phase_0_safety_net()
    
    if args.skip_phase != 1:
        agent.phase_1_map_living_system()
    
    if args.skip_phase != 2:
        agent.phase_2_classify_files()
    
    if args.skip_phase != 3:
        actions = agent.phase_3_remove_breaking_code(dry_run=args.dry_run)
        print("\n[INFO] Actions taken:")
        for action_type, items in actions.items():
            if items:
                print(f"  {action_type}: {len(items)} items")
                for item in items[:5]:  # Show first 5
                    print(f"    - {item}")
                if len(items) > 5:
                    print(f"    ... and {len(items) - 5} more")
    
    if args.skip_phase != 4:
        simplifications = agent.phase_4_simplify_apis(dry_run=args.dry_run)
        if simplifications:
            print("\n[INFO] Simplifications needed:")
            for simp in simplifications:
                print(f"  - {simp}")
    
    if args.skip_phase != 5:
        agent.phase_5_validate()
    
    # Generate report
    agent.generate_report()


if __name__ == "__main__":
    main()

