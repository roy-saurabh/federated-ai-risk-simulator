# 🚀 Deployment Guide - Federated AI Risk Simulator

## Free Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)

1. **Create GitHub Repository**
   ```bash
   # Initialize git repository
   git init
   git add .
   git commit -m "Initial commit: Federated AI Risk Simulator"
   
   # Create GitHub repository and push
   git remote add origin https://github.com/YOUR_USERNAME/federated-ai-risk-simulator.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `federated-ai-risk-simulator`
   - Set main file path: `federated_ai_risk_simulator.py`
   - Click "Deploy"

3. **Access Your App**
   - Your app will be available at: `https://federated-ai-risk-simulator-XXXXX.streamlit.app`
   - Share this URL with others

### Option 2: Render (Alternative FREE option)

1. **Create render.yaml**
   ```yaml
   services:
     - type: web
       name: federated-ai-risk-simulator
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run federated_ai_risk_simulator.py --server.port $PORT --server.address 0.0.0.0
   ```

2. **Deploy on Render**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub
   - Click "New Web Service"
   - Connect your GitHub repository
   - Configure as above
   - Deploy

### Option 3: Railway (Another FREE option)

1. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will auto-detect Python and deploy

## 🎯 Quick Deployment Steps

### Step 1: Prepare Repository
```bash
# Navigate to project directory
cd fed_al_simulator

# Initialize git
git init
git add .
git commit -m "Initial commit: Federated AI Risk Simulator"

# Create GitHub repository (do this on GitHub.com)
# Then push your code
git remote add origin https://github.com/YOUR_USERNAME/federated-ai-risk-simulator.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Fill in:
   - **Repository**: `YOUR_USERNAME/federated-ai-risk-simulator`
   - **Branch**: `main`
   - **Main file path**: `federated_ai_risk_simulator.py`
5. Click "Deploy"

### Step 3: Share Your App
- Your app will be live at: `https://federated-ai-risk-simulator-XXXXX.streamlit.app`
- Share this URL for public access

## 🔧 Configuration Files

### .streamlit/config.toml
```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

### requirements.txt
```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.15.0
kaleido>=0.2.1
```

## 📊 Deployment Checklist

- [ ] All files committed to GitHub
- [ ] requirements.txt includes all dependencies
- [ ] .streamlit/config.toml configured
- [ ] Main file path correctly specified
- [ ] App successfully deployed
- [ ] URL accessible and working
- [ ] All features functional in deployed version

## 🚨 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies in requirements.txt
2. **Port Issues**: Streamlit Cloud handles this automatically
3. **File Path Issues**: Use relative paths in code
4. **Memory Issues**: Optimize for cloud deployment

### Performance Tips
- Reduce default sample sizes for faster loading
- Optimize plot generation for cloud environment
- Use efficient data structures

## 🌐 Public URL

Once deployed, your app will be accessible at:
```
https://federated-ai-risk-simulator-XXXXX.streamlit.app
```

Replace `XXXXX` with your actual deployment ID.

## 📞 Support

For deployment issues:
- Check Streamlit Cloud documentation
- Review GitHub repository setup
- Verify all dependencies are included
- Test locally before deploying 