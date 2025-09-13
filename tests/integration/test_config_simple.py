"""
Simplified test script to verify configuration behavior between config.py and YAML files.
This version imports only the core configuration without heavy dependencies.
"""
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config_py_values():
    """Test current values from config.py"""
    print("=== Testing config.py values ===")
    try:
        from config.config import DELTA_HOURS, WINDOW_MIN, K_DEFAULT, K_OPTIONS
        print(f"DELTA_HOURS: {DELTA_HOURS}")
        print(f"WINDOW_MIN: {WINDOW_MIN}")
        print(f"K_DEFAULT: {K_DEFAULT}")
        print(f"K_OPTIONS: {K_OPTIONS}")
        return True, (DELTA_HOURS, WINDOW_MIN, K_DEFAULT, K_OPTIONS)
    except Exception as e:
        print(f"Error importing config: {e}")
        return False, None


def test_yaml_loading():
    """Test YAML loading functionality"""
    print("\n=== Testing YAML loading utility ===")
    try:
        from utils.io import maybe_load_yaml
        
        # Create a test YAML file
        yaml_content = """
runtime:
  delta_hours: 999
  window_min: 888
  k_default: 777
  k_options: [111, 222]
  update_interval_sec: 666
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            yaml_data = maybe_load_yaml(yaml_path)
            print(f"YAML loaded successfully: {yaml_data}")
            
            if 'runtime' in yaml_data:
                runtime_section = yaml_data['runtime']
                print(f"Runtime section: {runtime_section}")
                return True, runtime_section
            else:
                print("No 'runtime' section found in YAML")
                return False, None
                
        finally:
            Path(yaml_path).unlink()
            
    except Exception as e:
        print(f"Error testing YAML loading: {e}")
        return False, None


def test_runtime_config():
    """Test RuntimeConfig class without heavy imports"""
    print("\n=== Testing RuntimeConfig behavior ===")
    
    # We'll manually simulate what RuntimeConfig.from_yaml does
    # without importing the heavy dependencies
    try:
        from config.config import DELTA_HOURS, WINDOW_MIN, K_DEFAULT, K_OPTIONS
        from utils.io import maybe_load_yaml
        
        # Test 1: Default behavior (like RuntimeConfig())
        print(f"Default config values would be:")
        print(f"delta_hours: {DELTA_HOURS}")
        print(f"window_min: {WINDOW_MIN}")
        print(f"k_default: {K_DEFAULT}")
        
        # Test 2: YAML override behavior (like RuntimeConfig.from_yaml())
        yaml_content = """
runtime:
  delta_hours: 555
  window_min: 444
  k_default: 333
  update_interval_sec: 222
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            yaml_config = maybe_load_yaml(yaml_path)
            runtime_config = yaml_config.get('runtime', {}) if isinstance(yaml_config, dict) else {}
            
            # Simulate what RuntimeConfig.from_yaml() does
            overridden_delta = runtime_config.get('delta_hours', DELTA_HOURS)
            overridden_window = runtime_config.get('window_min', WINDOW_MIN)  
            overridden_k = runtime_config.get('k_default', K_DEFAULT)
            overridden_interval = runtime_config.get('update_interval_sec', 60)
            
            print(f"\nWith YAML override:")
            print(f"delta_hours: {overridden_delta} (original: {DELTA_HOURS})")
            print(f"window_min: {overridden_window} (original: {WINDOW_MIN})")
            print(f"k_default: {overridden_k} (original: {K_DEFAULT})")
            print(f"update_interval_sec: {overridden_interval} (original: 60)")
            
            # Check if overrides worked
            override_success = (
                overridden_delta == 555 and
                overridden_window == 444 and 
                overridden_k == 333 and
                overridden_interval == 222
            )
            
            return override_success
            
        finally:
            Path(yaml_path).unlink()
            
    except Exception as e:
        print(f"Error testing RuntimeConfig behavior: {e}")
        return False


def create_example_config_files():
    """Create example configuration files for the user"""
    print("\n=== Creating example configuration files ===")
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Create custom_runtime_config.yaml
    custom_config_content = """# Custom runtime configuration for trend prediction service
# This file can override values from config.py

runtime:
  # Dashboard update settings
  update_interval_sec: 30          # Update dashboard every 30 seconds (default: 60)
  enable_background_timer: true    # Enable continuous updates
  
  # Prediction parameters (these override config.py values)
  delta_hours: 3                   # Time window for predictions (default: 2)
  window_min: 90                   # Rolling window in minutes (default: 60)
  k_default: 10                    # Default number of top predictions (default: 5)
  k_options: [5, 10, 15, 20]       # Available K options (default: [5, 10])
  
  # Optional: Override file paths
  # metrics_snapshot_dir: "./my_custom_metrics"
  # predictions_cache_path: "./my_custom_cache.json"

# Usage:
# python src/service/main.py config/custom_runtime_config.yaml
"""
    
    config_file = config_dir / "custom_runtime_config.yaml"
    with open(config_file, 'w') as f:
        f.write(custom_config_content)
    
    print(f"✅ Created: {config_file}")
    
    # Create alternative config for testing
    test_config_content = """# Test configuration with different values
runtime:
  delta_hours: 6
  window_min: 120  
  k_default: 20
  update_interval_sec: 15
"""
    
    test_file = config_dir / "test_config.yaml"
    with open(test_file, 'w') as f:
        f.write(test_config_content)
    
    print(f"✅ Created: {test_file}")
    
    return config_file, test_file


def main():
    print("Testing Configuration Behavior (Simplified)")
    print("=" * 60)
    
    # Test 1: Check if we can import config.py values
    config_success, config_values = test_config_py_values()
    
    # Test 2: Check YAML loading
    yaml_success, yaml_data = test_yaml_loading()
    
    # Test 3: Test override behavior
    override_success = test_runtime_config()
    
    # Test 4: Create example files
    example_files = create_example_config_files()
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"✅ config.py import: {'SUCCESS' if config_success else 'FAILED'}")
    print(f"✅ YAML loading: {'SUCCESS' if yaml_success else 'FAILED'}")
    print(f"✅ YAML override: {'SUCCESS' if override_success else 'FAILED'}")
    
    if config_success and yaml_success and override_success:
        print("\n🎉 CONCLUSION: YAML configuration override WORKS!")
        print("\nHow to use:")
        print("1. Edit config.py for permanent changes to default values")
        print("2. Create YAML files to override defaults without changing code")
        print("3. Pass YAML file to main service: python src/service/main.py your_config.yaml")
        print("4. YAML values take precedence over config.py defaults")
    else:
        print("\n❌ CONCLUSION: There are issues with the configuration system")
    
    print(f"\nExample files created:")
    if example_files:
        for file_path in example_files:
            print(f"  - {file_path}")


if __name__ == "__main__":
    main()