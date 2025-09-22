# AI Provider Setup Guide

This guide helps you set up free and affordable AI providers for the Code Analyzer backend.

## 🚀 Quick Setup - Google Gemini (Recommended - FREE)

Google Gemini 1.5 Flash is **completely free** with generous quotas, perfect for development and testing.

### Step 1: Get Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API key"
3. Copy your API key

### Step 2: Configure Environment
Update your `.env` file:
```bash
# Set Gemini as your AI provider
AI_PROVIDER=gemini
AI_MODEL=gemini-1.5-flash

# Add your Gemini API key
GEMINI_API_KEY="your_actual_api_key_here"
```

### Step 3: Test
Restart your server and test the Q&A functionality!

---

## 🔧 Alternative Providers

### DeepSeek Coder (Excellent for Code Analysis)
- **Cost**: Very affordable (~$0.14 per 1M tokens)
- **Specialty**: Optimized for coding tasks
- **Setup**: Get API key from [DeepSeek Platform](https://platform.deepseek.com/)

```bash
AI_PROVIDER=deepseek
AI_MODEL=deepseek-coder
DEEPSEEK_API_KEY="your_deepseek_api_key"
```

### Ollama (Local Models - FREE)
- **Cost**: Completely free (runs locally)
- **Models**: DeepSeek Coder, CodeLlama, Llama 3.1, etc.
- **Setup**: 
  1. Install [Ollama](https://ollama.ai/)
  2. Run: `ollama pull deepseek-coder:6.7b`

```bash
AI_PROVIDER=ollama
AI_MODEL=deepseek-coder:6.7b
OLLAMA_BASE_URL=http://localhost:11434
```

### Anthropic Claude 3.5 Haiku
- **Cost**: Affordable (~$0.25 per 1M tokens)
- **Specialty**: Excellent reasoning and code understanding
- **Setup**: Get API key from [Anthropic Console](https://console.anthropic.com/)

```bash
AI_PROVIDER=anthropic
AI_MODEL=claude-3-5-haiku-20241022
ANTHROPIC_API_KEY="your_anthropic_api_key"
```

---

## 🎯 Recommended Setup for Different Use Cases

### For Development/Testing
```bash
AI_PROVIDER=gemini
AI_MODEL=gemini-1.5-flash
```
- **Why**: Free, fast, good quality
- **Quota**: 15 requests/minute, 1M requests/day

### For Production (Cost-Effective)
```bash
AI_PROVIDER=deepseek
AI_MODEL=deepseek-coder
```
- **Why**: Specialized for code, very affordable
- **Best for**: Code analysis and programming Q&A

### For Privacy-Conscious Users
```bash
AI_PROVIDER=ollama
AI_MODEL=deepseek-coder:6.7b
```
- **Why**: Runs completely locally, no data sent to external APIs
- **Requirements**: 8GB+ RAM for 6.7B model

---

## 🔍 Testing Your Setup

After configuring your provider, test it:

```bash
# Run the PDF test script
.\test_pdf.ps1

# Or test Q&A directly
curl -X POST "http://127.0.0.1:8001/api/v1/analysis/qa" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this code about?", "analysis_id": "your_analysis_id"}'
```

---

## 📊 Provider Comparison

| Provider | Cost | Speed | Code Quality | Setup Difficulty |
|----------|------|-------|--------------|------------------|
| **Gemini 1.5 Flash** | FREE ⭐ | Fast | Very Good | Easy |
| **DeepSeek Coder** | Very Low | Fast | Excellent | Easy |
| **Ollama (Local)** | FREE ⭐ | Medium | Good | Medium |
| **Claude 3.5 Haiku** | Low | Fast | Excellent | Easy |
| **OpenAI GPT-3.5** | Medium | Fast | Good | Easy |

---

## 🛠️ Troubleshooting

### "Provider not configured" error
- Make sure you've set the correct API key in `.env`
- Restart your server after changing `.env`

### Gemini "API key not valid" error
- Verify your API key at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Make sure you've enabled the Gemini API

### Ollama connection failed
- Install Ollama from [ollama.ai](https://ollama.ai/)
- Run `ollama serve` to start the server
- Pull your model: `ollama pull deepseek-coder:6.7b`

### Rate limits
- Gemini: 15 requests/minute (free tier)
- DeepSeek: Higher limits with paid account
- Ollama: No limits (local)

---

## 💡 Pro Tips

1. **Start with Gemini** - It's free and works great for most use cases
2. **Use DeepSeek for code-heavy workloads** - It's optimized for programming tasks
3. **Try Ollama for privacy** - Everything runs locally
4. **Monitor your usage** - Most providers have dashboards to track API usage

