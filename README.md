# 🔬 AI Research Pipeline

Turn any topic into a fully cited, structured research report — powered by a multi-agent system that searches the live web, validates sources, extracts evidence, and writes with inline citations.

![Streamlit UI](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit)
![CrewAI](https://img.shields.io/badge/Orchestration-CrewAI-000000?logo=crewai)
![Groq](https://img.shields.io/badge/LLM-Groq-F55036?logo=groq)

---

## ✨ What it does

Type a research topic and click **Run**. Within ~90 seconds you get:

- A **Markdown report** with Executive Summary, Background, Methods, Benchmarks, Key Findings, Open Problems, and Recommendations
- **Inline citations** `[1]`, `[2]` linking to real, verified sources
- A **downloadable `.md` file** ready for Notion, Google Docs, or LaTeX
- Optional **debug JSON** showing every step from query generation to final synthesis

---

## 🏗️ Architecture

The system runs as a **team of 4 specialized AI agents**, each with a single job:

| Agent | Role | What it does |
|---|---|---|
| **Planner** | Research Strategist | Breaks your topic into 5 orthogonal search angles (foundations, methods, benchmarks, applications, open problems) |
| **Validator** | Source Librarian | Scores web results on credibility and picks the top 4 (arXiv, IEEE, .edu ranked highest) |
| **Extractor** | Evidence Reader | Fetches full pages, pulls verbatim quotes, exact metrics, and dataset names |
| **Writer** | Senior Analyst | Synthesizes everything into a structured Markdown report, citing only the extracted evidence |

**Search & Fetch are deterministic code** — not hallucinated by the LLM. The agents only touch what the web actually returned.

---

## 🚀 Quickstart

### 1. Clone & enter the folder
```bash
git clone https://github.com/yourusername/research-pipeline.git
cd research-pipeline
```

### 2. Create a virtual environment
```bash
python -m venv venvv
# Windows
venvv\Scripts\activate
# macOS/Linux
source venvv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys
```bash
# Copy the example file
cp .env.example .env
```
Edit `.env` and paste your keys:
```
GROQ_API_KEY=gsk_your_key_here
TAVILY_API_KEY=tvly_your_key_here
```
- [Get a free Groq key](https://console.groq.com/keys)
- [Get a free Tavily key](https://app.tavily.com/)

### 5. Launch
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 📸 Interface

```
🔬 AI Research Pipeline
Turn any topic into a fully cited research report — powered by AI

[ Enter research topic                                    ]
[ Run Research Pipeline                                   ]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  100%
✅ All done!

Research Report
─────────────────
# Executive Summary
...
```

---

## 🎯 Why this is more accurate than ChatGPT, Perplexity, or a single LLM

| Capability | Single LLM (temp=0) | ChatGPT Browsing | Perplexity | **This Pipeline** |
|---|---|---|---|---|
| **Citations** | Often hallucinated | Black-box, unverifiable | Prettified but un-auditable | **Every claim tied to a validated URL via Extractor** |
| **Source freshness** | Training cutoff only | Real-time but opaque | Real-time but opaque | **Live Tavily search + domain scoring** |
| **Evidence chain** | None | None | None | **JSON audit trail: queries → sources → extracted quotes → report** |
| **Can say "insufficient evidence"** | No — invents content | No — rounds confidently | No — optimizes for engagement | **Yes — Writer is firewalled from the internet** |
| **Cost at scale** | $$$ (OpenAI GPT-4) | $$$ (subscription) | $$$ (subscription) | **Free tier: Groq + Tavily** |

> **The critical difference:** A single LLM at `temperature=0` is a *confident liar with a fixed seed*. It will give you the same fake citation every time. This pipeline forces the LLM to **earn every sentence** from live, extracted evidence — and allows it to remain silent when the evidence is not there.

---

## 🛠️ Tech Stack

- **[CrewAI](https://crewai.com)** — Multi-agent orchestration with role-based task separation
- **[Groq](https://groq.com)** — `llama-3.3-70b-versatile` via LiteLLM (12K TPM free tier)
- **[Tavily](https://tavily.com)** — Academic web search API (1,000 searches/month free)
- **[Streamlit](https://streamlit.io)** — Frontend UI
- **BeautifulSoup + pdfplumber** — Deterministic HTML/PDF text extraction
- **Pydantic v2** — Structured output validation at every stage

---

## 📁 Project Structure

```
research-pipeline/
├── app.py                 # Streamlit frontend
├── agents.py              # CrewAI backend — 4 agents + pipeline logic
├── requirements.txt       # Python dependencies
├── .env.example           # API key template (safe to commit)
├── .gitignore             # Secrets + cache exclusions
├── outputs/               # Generated reports (optional)
└── README.md              # You are here
```

---

## 📝 Example Topics

Try these for rich, fully cited reports:

- `Retrieval-Augmented Generation evaluation benchmarks and failure modes in enterprise knowledge bases`
- `Chain-of-Thought prompting and automatic reasoning optimization: GSM8K and MATH benchmarks`
- `Diffusion models for text-to-image generation: Stable Diffusion vs DALL-E 3 vs Imagen`

---

## ⚙️ Configuration

Edit `.env` to customize:

```bash
# Optional: override the default model
# Available on Groq free tier: llama-3.3-70b-versatile, llama-4-scout, gpt-oss-20b, qwen-3-32b
GROQ_MODEL=llama-3.3-70b-versatile
```

---

## 🐛 Debugging

Enable **"Show behind-the-scenes details (debug)"** in the sidebar to inspect:

- Generated search queries
- Raw search results with domain scores
- Validated source rankings
- Extracted evidence, datasets, metrics, and verbatim quotes
- Full timestamps per stage

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-idea`)
3. Commit your changes (`git commit -m 'Add amazing idea'`)
4. Push to the branch (`git push origin feature/amazing-idea`)
5. Open a Pull Request

---

## 📄 License

MIT License — free for personal, academic, and commercial use.

---

## 🙋 Support

If the pipeline fails, check the red error message in the UI. Common fixes:

- **"Missing environment variables"** → Your `.env` file is missing or in the wrong folder
- **"Rate limit reached"** → Groq's free tier hit its 12K tokens/minute cap; wait 30 seconds and retry
- **"No search results"** → The topic may be too niche; try a broader phrasing
