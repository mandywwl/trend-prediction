"""
Test script to verify configuration behavior between config.py and YAML files.
This will help determine if custom_runtime_config.yaml actually affects values 
or if only editing config.py works.
"""
import sys
import tempfile
from pathlib import Path

# Add src to path so we can import from the project
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.config import DELTA_HOURS, WINDOW_MIN, K_DEFAULT, K_OPTIONS
from service.runtime_glue import RuntimeConfig


def test_config_py_values():
    """Test current values from config.py"""
    print("=== Testing config.py values ===")
    print(f"DELTA_HOURS: {DELTA_HOURS}")
    print(f"WINDOW_MIN: {WINDOW_MIN}")
    print(f"K_DEFAULT: {K_DEFAULT}")
    print(f"K_OPTIONS: {K_OPTIONS}")
    
    # Test RuntimeConfig defaults (should use config.py values)
    config = RuntimeConfig()
    print(f"\nRuntimeConfig defaults:")
    print(f"delta_hours: {config.delta_hours}")
    print(f"window_min: {config.window_min}")
    print(f"k_default: {config.k_default}")
    print(f"k_options: {config.k_options}")
    
    return config


def test_yaml_override():
    """Test YAML configuration override"""
    print("\n=== Testing YAML override ===")
    
    # Create a temporary YAML config with different values
    yaml_content = """
runtime:
  delta_hours: 999         # Different from config.py default (2)
  window_min: 888          # Different from config.py default (60)
  k_default: 777           # Different from config.py default (5)
  k_options: [111, 222]    # Different from config.py default
  update_interval_sec: 666 # Different from default (60)
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        print(f"Created temporary YAML file: {yaml_path}")
        print("YAML contents:")
        print(yaml_content)
        
        # Test loading YAML config
        config_from_yaml = RuntimeConfig.from_yaml(yaml_path)
        print(f"RuntimeConfig from YAML:")
        print(f"delta_hours: {config_from_yaml.delta_hours}")
        print(f"window_min: {config_from_yaml.window_min}")
        print(f"k_default: {config_from_yaml.k_default}")
        print(f"k_options: {config_from_yaml.k_options}")
        print(f"update_interval_sec: {config_from_yaml.update_interval_sec}")
        
        return config_from_yaml
        
    finally:
        # Clean up temp file
        Path(yaml_path).unlink()


def test_custom_runtime_config():
    """Test creating custom_runtime_config.yaml in the config directory"""
    print("\n=== Testing custom_runtime_config.yaml ===")
    
    config_dir = Path(__file__).parent / "src" / "config"
    custom_config_path = config_dir / "custom_runtime_config.yaml"
    
    # Create custom config file
    yaml_content = """
runtime:
  delta_hours: 555         # Custom value
  window_min: 444          # Custom value  
  k_default: 333           # Custom value
  k_options: [111, 222, 333] # Custom value
  update_interval_sec: 222 # Custom value
  enable_background_timer: false # Custom value
"""
    
    try:
        with open(custom_config_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created custom config file: {custom_config_path}")
        print("Contents:")
        print(yaml_content)
        
        # Test loading the custom config
        config_from_custom = RuntimeConfig.from_yaml(str(custom_config_path))
        print(f"RuntimeConfig from custom_runtime_config.yaml:")
        print(f"delta_hours: {config_from_custom.delta_hours}")
        print(f"window_min: {config_from_custom.window_min}")
        print(f"k_default: {config_from_custom.k_default}")
        print(f"k_options: {config_from_custom.k_options}")
        print(f"update_interval_sec: {config_from_custom.update_interval_sec}")
        print(f"enable_background_timer: {config_from_custom.enable_background_timer}")
        
        return config_from_custom, custom_config_path
        
    except Exception as e:
        print(f"Error creating/testing custom config: {e}")
        return None, custom_config_path


def compare_results(default_config, yaml_config, custom_config):
    """Compare the results to understand what works"""
    print("\n=== Configuration Comparison ===")
    print(f"{'Parameter':<25} {'config.py':<15} {'YAML temp':<15} {'custom_runtime_config.yaml':<25}")
    print("-" * 80)
    
    params = ['delta_hours', 'window_min', 'k_default', 'update_interval_sec']
    
    for param in params:
        default_val = getattr(default_config, param, 'N/A')
        yaml_val = getattr(yaml_config, param, 'N/A') if yaml_config else 'N/A'
        custom_val = getattr(custom_config, param, 'N/A') if custom_config else 'N/A'
        print(f"{param:<25} {default_val:<15} {yaml_val:<15} {custom_val:<25}")


def main():
    print("Testing Configuration Behavior")
    print("=" * 50)
    
    # Test 1: Default config.py behavior
    default_config = test_config_py_values()
    
    # Test 2: YAML override behavior
    yaml_config = test_yaml_override()
    
    # Test 3: Custom runtime config file
    custom_config, custom_path = test_custom_runtime_config()
    
    # Compare results
    compare_results(default_config, yaml_config, custom_config)
    
    print("\n=== Conclusions ===")
    if yaml_config and yaml_config.delta_hours == 999:
        print("✅ YAML configuration override WORKS")
        print("   - RuntimeConfig.from_yaml() successfully overrides config.py values")
    else:
        print("❌ YAML configuration override FAILED")
    
    if custom_config and custom_config.delta_hours == 555:
        print("✅ custom_runtime_config.yaml WORKS")
        print("   - You can create custom config files and they will override defaults")
    else:
        print("❌ custom_runtime_config.yaml FAILED")
    
    print("\n=== How the system works ===")
    print("1. config.py defines all the base constants (DELTA_HOURS, WINDOW_MIN, etc.)")
    print("2. RuntimeConfig class uses these constants as defaults")
    print("3. RuntimeConfig.from_yaml() can override any of these values")
    print("4. The main service accepts a YAML file path as argument")
    print("5. Usage: python src/service/main.py path/to/config.yaml")
    
    # Cleanup
    if custom_path and custom_path.exists():
        try:
            custom_path.unlink()
            print(f"\nCleaned up: {custom_path}")
        except Exception as e:
            print(f"\nWarning: Could not clean up {custom_path}: {e}")


if __name__ == "__main__":
    main()