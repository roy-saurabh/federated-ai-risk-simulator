#!/bin/bash

echo "🚀 Federated AI Risk Simulator - Deployment Script"
echo "=================================================="
echo

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "federated_ai_risk_simulator.py" ]; then
    echo "❌ Please run this script from the fed_al_simulator directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "✅ Git found"
echo "✅ In correct directory"
echo

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files
echo "Adding files to git..."
git add .
echo "✅ Files added"

# Commit changes
echo "Committing changes..."
git commit -m "Deploy Federated AI Risk Simulator - $(date)"
echo "✅ Changes committed"

echo
echo "🎯 Next Steps for Deployment:"
echo "============================="
echo
echo "1. Create a GitHub repository:"
echo "   - Go to https://github.com/new"
echo "   - Name it: federated-ai-risk-simulator"
echo "   - Make it public"
echo "   - Don't initialize with README (we already have one)"
echo
echo "2. Connect and push to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/federated-ai-risk-simulator.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo
echo "3. Deploy on Streamlit Cloud:"
echo "   - Go to https://share.streamlit.io"
echo "   - Sign in with GitHub"
echo "   - Click 'New app'"
echo "   - Repository: YOUR_USERNAME/federated-ai-risk-simulator"
echo "   - Branch: main"
echo "   - Main file path: federated_ai_risk_simulator.py"
echo "   - Click 'Deploy'"
echo
echo "4. Share your app:"
echo "   Your app will be available at: https://federated-ai-risk-simulator-XXXXX.streamlit.app"
echo
echo "📚 For detailed instructions, see DEPLOYMENT.md"
echo
echo "🔗 Quick Links:"
echo "   - GitHub: https://github.com/new"
echo "   - Streamlit Cloud: https://share.streamlit.io"
echo "   - Deployment Guide: DEPLOYMENT.md" 