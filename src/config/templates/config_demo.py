"""
Practical demonstration of config.py vs YAML configuration behavior.
This script shows exactly which method affects values and how.
"""
import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demonstrate_config_behavior():
    """Demonstrate how both config.py and YAML work together"""
    print("Configuration Behavior Demonstration")
    print("=" * 50)
    
    # Import the config values and utilities
    from config.config import DELTA_HOURS, WINDOW_MIN, K_DEFAULT, K_OPTIONS
    from utils.io import maybe_load_yaml
    
    print("STEP 1: Values from config.py (base defaults)")
    print(f"  DELTA_HOURS = {DELTA_HOURS}")
    print(f"  WINDOW_MIN = {WINDOW_MIN}")
    print(f"  K_DEFAULT = {K_DEFAULT}")
    print(f"  K_OPTIONS = {K_OPTIONS}")
    
    print("\nSTEP 2: How RuntimeConfig uses these values")
    print("  When you create RuntimeConfig() without YAML:")
    print(f"    config.delta_hours = {DELTA_HOURS}  # Uses config.py")
    print(f"    config.window_min = {WINDOW_MIN}   # Uses config.py")
    print(f"    config.k_default = {K_DEFAULT}     # Uses config.py")
    
    print("\nSTEP 3: How YAML overrides work")
    
    # Create test YAML with custom values
    yaml_content = """
runtime:
  delta_hours: 8        # Override config.py value (was 2)
  window_min: 180       # Override config.py value (was 60)
  k_default: 15         # Override config.py value (was 5)
  update_interval_sec: 45  # This doesn't exist in config.py, uses YAML value
"""
    
    print("  Example YAML file:")
    print("  " + yaml_content.replace("\n", "\n  "))
    
    # Simulate what RuntimeConfig.from_yaml() does
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        yaml_config = maybe_load_yaml(yaml_path)
        runtime_section = yaml_config.get('runtime', {})
        
        # This is what RuntimeConfig.from_yaml() actually does:
        final_delta = runtime_section.get('delta_hours', DELTA_HOURS)
        final_window = runtime_section.get('window_min', WINDOW_MIN) 
        final_k = runtime_section.get('k_default', K_DEFAULT)
        final_interval = runtime_section.get('update_interval_sec', 60)  # 60 is hardcoded default
        
        print("\n  Result after YAML override:")
        print(f"    config.delta_hours = {final_delta}       # YAML overrode config.py ({DELTA_HOURS} → {final_delta})")
        print(f"    config.window_min = {final_window}      # YAML overrode config.py ({WINDOW_MIN} → {final_window})")
        print(f"    config.k_default = {final_k}        # YAML overrode config.py ({K_DEFAULT} → {final_k})")
        print(f"    config.update_interval_sec = {final_interval}  # YAML provided new value")
        
    finally:
        Path(yaml_path).unlink()
    
    print("\nSTEP 4: Practical usage patterns")
    print("  Method 1 - Editing config.py:")
    print("    ✅ Changes affect ALL runs of the application")
    print("    ✅ Permanent changes, committed to git")
    print("    ❌ Requires code changes for different environments")
    
    print("\n  Method 2 - Using YAML files:")  
    print("    ✅ Can have different configs for different environments")
    print("    ✅ No code changes needed")
    print("    ✅ Can override only specific values")
    print("    ✅ Falls back to config.py defaults for unspecified values")


def show_practical_examples():
    """Show practical examples of both approaches"""
    print("\n" + "=" * 50)
    print("PRACTICAL EXAMPLES")
    print("=" * 50)
    
    print("Example 1: Development vs Production")
    print("  config.py (shared defaults):")
    print("    DELTA_HOURS = 2")
    print("    WINDOW_MIN = 60")
    print("    K_DEFAULT = 5")
    
    print("\n  dev_config.yaml (faster updates for development):")
    print("    runtime:")
    print("      update_interval_sec: 10  # Update every 10 seconds")
    print("      delta_hours: 1           # Shorter prediction window")
    
    print("\n  prod_config.yaml (production optimized):")
    print("    runtime:")
    print("      update_interval_sec: 300 # Update every 5 minutes") 
    print("      k_default: 10            # More predictions")
    print("      window_min: 120          # Longer analysis window")
    
    print("\nExample 2: A/B Testing configurations")
    print("  config_variant_a.yaml:")
    print("    runtime:")
    print("      k_default: 5")
    print("      delta_hours: 2")
    
    print("\n  config_variant_b.yaml:")
    print("    runtime:")
    print("      k_default: 10") 
    print("      delta_hours: 4")
    
    print("\nExample 3: Quick testing with custom values")
    print("  test_config.yaml:")
    print("    runtime:")
    print("      delta_hours: 24          # Test daily predictions")
    print("      window_min: 1440         # 24-hour window")
    print("      update_interval_sec: 5   # Fast updates for testing")


def show_usage_instructions():
    """Show how to actually use the configuration system"""
    print("\n" + "=" * 50)
    print("HOW TO USE")
    print("=" * 50)
    
    print("1. To use config.py values (defaults):")
    print("   python src/service/main.py")
    print("   # Uses all values from config.py")
    
    print("\n2. To override with YAML:")
    print("   python src/service/main.py config/custom_runtime_config.yaml")
    print("   # Overrides values specified in YAML, uses config.py for others")
    
    print("\n3. To create your own config:")
    config_example = """   # Create my_config.yaml:
   runtime:
     delta_hours: 6
     window_min: 120
     k_default: 8
     update_interval_sec: 30
   
   # Then run:
   python src/service/main.py my_config.yaml"""
    print(config_example)
    
    print("\n4. Current config files available:")
    config_dir = Path("config")
    src_config_dir = Path("src/config")
    
    yaml_files = []
    for directory in [config_dir, src_config_dir]:
        if directory.exists():
            yaml_files.extend(directory.glob("*.yaml"))
    
    template_dir = Path("src/config/templates")
    if template_dir.exists():
        yaml_files.extend(template_dir.glob("*.yaml"))
    
    for yaml_file in yaml_files:
        print(f"   - {yaml_file}")
        if yaml_file.name == "custom_runtime_config.yaml":
            print("     (Created by our test - ready to use!)")


def main():
    """Main demonstration"""
    try:
        demonstrate_config_behavior()
        show_practical_examples()
        show_usage_instructions()
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print("✅ BOTH methods work:")
        print("   • config.py: Defines default values for all parameters")
        print("   • YAML files: Override specific values without changing code")
        print("   • YAML takes precedence when both are present")
        print("   • Unspecified YAML values fall back to config.py defaults")
        
        print("\n🎯 RECOMMENDATION:")
        print("   • Use config.py for stable, shared defaults")
        print("   • Use YAML files for environment-specific overrides")
        print("   • Pass YAML file path as command line argument to main.py")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())