#!/usr/bin/env python3
"""
Federated AI Risk Simulator Launcher
====================================

Simple launcher script that checks dependencies and runs the simulator.
"""

import sys
import subprocess
import importlib.util

def check_dependency(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        return False
    return True

def install_dependency(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🤖 Federated AI Risk Simulator Launcher")
    print("=" * 50)
    
    # Check dependencies
    dependencies = [
        ("streamlit", "streamlit"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("plotly", "plotly"),
        ("kaleido", "kaleido")
    ]
    
    missing_deps = []
    
    print("Checking dependencies...")
    for package_name, import_name in dependencies:
        if check_dependency(package_name, import_name):
            print(f"✅ {package_name}")
        else:
            print(f"❌ {package_name} (missing)")
            missing_deps.append(package_name)
    
    if missing_deps:
        print(f"\nInstalling missing dependencies: {', '.join(missing_deps)}")
        for dep in missing_deps:
            print(f"Installing {dep}...")
            if install_dependency(dep):
                print(f"✅ {dep} installed successfully")
            else:
                print(f"❌ Failed to install {dep}")
                print("Please install manually: pip install -r requirements.txt")
                return
    
    print("\n🚀 Starting Federated AI Risk Simulator...")
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.")
    
    try:
        # Run the Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "federated_ai_risk_simulator.py"])
    except KeyboardInterrupt:
        print("\n👋 Simulator stopped. Thanks for using the Federated AI Risk Simulator!")
    except FileNotFoundError:
        print("❌ Error: federated_ai_risk_simulator.py not found in current directory")
        print("Please make sure you're running this script from the project directory.")
    except Exception as e:
        print(f"❌ Error starting simulator: {e}")
        print("Try running manually: streamlit run federated_ai_risk_simulator.py")

if __name__ == "__main__":
    main() 