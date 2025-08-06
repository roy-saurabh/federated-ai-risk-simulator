#!/bin/bash

echo "🚀 Federated AI Risk Simulator - Deployment Setup"
echo "================================================="
echo

# Get current git status
echo "📊 Current Git Status:"
git status --porcelain
echo

# Check if we have commits
if git log --oneline -1 > /dev/null 2>&1; then
    echo "✅ Git repository is ready with commits"
else
    echo "❌ No commits found. Making initial commit..."
    git add .
    git commit -m "Initial commit: Federated AI Risk Simulator"
fi

echo
echo "🎯 Next Steps for Deployment:"
echo "============================="
echo
echo "1. Create a NEW GitHub repository:"
echo "   - Go to: https://github.com/new"
echo "   - Repository name: federated-ai-risk-simulator"
echo "   - Make it PUBLIC (required for free hosting)"
echo "   - DO NOT initialize with README (we already have one)"
echo "   - Click 'Create repository'"
echo
echo "2. After creating the repository, run these commands:"
echo "   (Replace YOUR_USERNAME with your actual GitHub username)"
echo
echo "   git remote add origin https://github.com/YOUR_USERNAME/federated-ai-risk-simulator.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo
echo "3. Deploy on Streamlit Cloud:"
echo "   - Go to: https://share.streamlit.io"
echo "   - Sign in with GitHub"
echo "   - Click 'New app'"
echo "   - Repository: YOUR_USERNAME/federated-ai-risk-simulator"
echo "   - Branch: main"
echo "   - Main file path: federated_ai_risk_simulator.py"
echo "   - Click 'Deploy'"
echo
echo "4. Your app will be live at:"
echo "   https://federated-ai-risk-simulator-XXXXX.streamlit.app"
echo
echo "🔗 Quick Links:"
echo "   - GitHub: https://github.com/new"
echo "   - Streamlit Cloud: https://share.streamlit.io"
echo
echo "💡 Tip: Make sure to use YOUR OWN GitHub username, not 'roy-saurabh'"
echo
echo "📚 For detailed instructions, see DEPLOYMENT.md" 