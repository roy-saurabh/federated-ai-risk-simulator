#!/bin/bash

echo "🤖 Federated AI Risk Simulator Launcher"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python 3 found"
echo

# Install dependencies if needed
echo "Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    echo "Please check your internet connection and try again"
    exit 1
fi

echo "✅ Dependencies installed"
echo

echo "🚀 Starting Federated AI Risk Simulator..."
echo "The application will open in your default web browser."
echo "If it doesn't open automatically, navigate to: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application."
echo

# Run the Streamlit app
streamlit run federated_ai_risk_simulator.py

echo
echo "👋 Simulator stopped. Thanks for using the Federated AI Risk Simulator!" 