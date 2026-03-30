"""
LeadIQ — Real Estate Intelligence Backend
FastAPI + OpenAI GPT-4o  (no pydantic — pure Python 3.14 compatible)
"""

import json
import os
from pathlib import Path
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
import anthropic
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("leadiq")

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# App + CORS
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LeadIQ — Real Estate Intelligence API",
    description="AI-powered lead scoring & chatbot for real estate boutiques.",
    version="1.0.0",
)

# ──────────────────────────────────────────────────────────────────────────────
# Static Files & Frontend Path
# ──────────────────────────────────────────────────────────────────────────────
_DIR = Path(__file__).parent
FRONTEND_PATH = _DIR.parent / "frontend"

if FRONTEND_PATH.exists():
    logger.info(f"Serving static files from: {FRONTEND_PATH}")
    app.mount("/assets", StaticFiles(directory=FRONTEND_PATH / "assets"), name="assets")
else:
    logger.warning(f"Frontend path NOT found: {FRONTEND_PATH}")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
DEFAULT_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
DEFAULT_GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash") # Switching to Flash as it's more widely available and faster


# ──────────────────────────────────────────────────────────────────────────────
# Unified Completion Service
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Provider Clients (cached)
# ──────────────────────────────────────────────────────────────────────────────

_clients = {}

def get_client(provider: str, api_key: str):
    cache_key = f"{provider}:{api_key}"
    if cache_key in _clients:
        return _clients[cache_key]
    
    logger.info(f"Initializing new client for provider: {provider}")
    if provider == "openai":
        client = AsyncOpenAI(api_key=api_key)
    elif provider == "anthropic":
        client = anthropic.AsyncAnthropic(api_key=api_key)
    elif provider == "google":
        genai.configure(api_key=api_key)
        client = genai
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    _clients[cache_key] = client
    return client

class CompletionService:
    def __init__(self, provider: str, api_key: str):
        self.provider = provider.lower()
        self.api_key = api_key

    async def get_completion(self, system_prompt: str, user_prompt: str, messages: Optional[list] = None, temperature: float = 0.7, json_mode: bool = False) -> str:
        logger.info(f"Completion requested for provider: {self.provider}")
        try:
            client = get_client(self.provider, self.api_key)
            if self.provider == "openai":
                args: dict[str, Any] = {
                    "model": DEFAULT_OPENAI_MODEL,
                    "temperature": temperature,
                    "messages": [{"role": "system", "content": system_prompt}] + (messages or [{"role": "user", "content": user_prompt}]),
                }
                if json_mode:
                    args["response_format"] = {"type": "json_object"}
                
                response = await client.chat.completions.create(**args)
                content = response.choices[0].message.content.strip()

            elif self.provider == "anthropic":
                anthropic_messages = messages or [{"role": "user", "content": user_prompt}]
                
                # Fix for Anthropic: Ensure first message is user
                if anthropic_messages and anthropic_messages[0]["role"] != "user":
                    anthropic_messages.insert(0, {"role": "user", "content": "Hello!"})
                    
                response = await client.messages.create(
                    model=DEFAULT_ANTHROPIC_MODEL,
                    max_tokens=2048,
                    temperature=temperature,
                    system=system_prompt,
                    messages=anthropic_messages
                )
                content = response.content[0].text.strip()

            elif self.provider == "google":
                # Use GenerativeModel with the corrected model name
                model = client.GenerativeModel(
                    model_name=DEFAULT_GOOGLE_MODEL,
                    system_instruction=system_prompt,
                    generation_config={
                        "temperature": temperature,
                        "response_mime_type": "application/json" if json_mode else "text/plain",
                    }
                )
                
                if messages:
                    # Map messages to Gemini history
                    history = []
                    
                    # Ensure first message is user
                    if messages[0]["role"] != "user":
                         history.append({"role": "user", "parts": ["Hello!"]})
                         
                    for m in messages[:-1]:
                        # Gemini roles are 'user' and 'model'
                        role = "user" if m["role"] == "user" else "model"
                        history.append({"role": role, "parts": [m["content"]]})
                    chat = model.start_chat(history=history)
                    last_msg = messages[-1]["content"] if messages else user_prompt
                    response = await chat.send_message_async(last_msg)
                else:
                    response = await model.generate_content_async(user_prompt)
                
                content = response.text.strip()
            
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported provider: {self.provider}")
                
            # Strip markdown blocks for JSON mode
            if json_mode:
                if content.startswith("```json"):
                    content = content[7:]
                elif content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
            return content
        except Exception as e:
            logger.error(f"Error in {self.provider} completion: {str(e)}")
            raise

def get_service(api_key: Optional[str], provider: Optional[str] = "openai") -> CompletionService:
    provider = provider or "openai"
    key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
    if not key:
        raise HTTPException(
            status_code=401,
            detail=f"No API key provided for {provider}. Add the X-API-Key header."
        )
    return CompletionService(provider, key)


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

ANALYZE_SYSTEM_PROMPT = """
[REQUIRED LOGIC: THE SCAM-FIRST FILTER]
You are now a Cynical Real Estate Broker with 30 years of experience. You have been scammed before and are highly suspicious.

YOUR PRIMARY GOAL: Identify "Too Good to be True" leads and flag them as [FRAUD/SPAM - LOW PRIORITY].

AUTOMATIC "LOW" TRIGGERS (OVERRIDE ALL OTHER SCORES):

"The Military/Overseas Trap": If a lead is "stationed," "overseas," or "on a mission" but wants to buy "Full Price" immediately. (STATUS: SCAM)

"The Bank Detail Request": If the first message mentions "Wiring money," "Bank details," or "Earnest money" via wire. (STATUS: PHISHING)

"The Scripted Greeting": "HELLO DEAR," "GOD BLESS," or excessive ALL CAPS combined with a high-value offer. (STATUS: SPAM)

MANDATORY RULE: A lead offering $2,000,000 in a "Wire Transfer" from "Overseas" is ZERO VALUE. Do not let the dollar amount distract you.

For each inquiry return a JSON object with EXACTLY these fields:
{
  "name": "<extracted name or 'Anonymous'>",
  "budget": "<extracted or inferred budget range, e.g. '$500K–$750K', '$1M+', 'Not stated'>",
  "urgency": "<'High' | 'Medium' | 'Low'>",
  "sentiment": "<'Excited' | 'Professional' | 'Frustrated' | 'Neutral'>",
  "priority_score": <integer 1–10>,
  "summary": "<one crisp sentence summarising this lead's intent>"
}

Urgency rules (follow strictly):
- High   → deadline within 30 days, or words like 'immediate', 'urgent', 'ASAP', 'this week'.
- Medium → planning 3–6 months out, or has a condition (e.g. 'selling my house first').
- Low    → browsing, no clear timeline, no financial commitment signals.

Priority Score rules:
- Start at 5.
- IF the lead triggers any of the AUTOMATIC "LOW" TRIGGERS above, they must be categorized as [FRAUD/SPAM - LOW PRIORITY] and the priority score MUST be 1.
- +3 if Urgency = High; +1 if Medium; -1 if Low.
- +1 if budget clearly stated and substantial (>$400K).
- +1 if sentiment = Excited or Professional.
- -1 if sentiment = Frustrated.
- Clamp final score between 1 and 10.

Return ONLY a valid JSON object with key "leads" containing an array (no markdown).
""".strip()

REPLY_SYSTEM_PROMPT = """
You are a top-producing real estate agent writing a personalised email reply.
Tone: warm for Excited, professional for Professional, empathetic for Frustrated, informative for Neutral.
Length: 150–250 words. Close with a clear call to action. Sign as "The LeadIQ Team".
Mention budget and urgency naturally — do not sound robotic.
Return ONLY the email body text (no subject line, no markdown).
""".strip()

CHATBOT_SYSTEM_PROMPT = """
You are Alex, a friendly and knowledgeable real estate assistant for a boutique property firm.
You are chatting with a buyer lead who is exploring their options (Medium or Low priority).
Goals:
1. Keep them engaged and interested.
2. Learn their needs — budget, preferred areas, property type, timeline.
3. Never be pushy. Be warm, helpful, and informative.
4. If they show signs of urgency, gently encourage a consultation booking.

Lead context: {lead_context}

Respond in English only. Keep replies to 2–4 sentences unless a detailed question is asked.
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_PATH / "index.html")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "LeadIQ API", "providers": ["openai", "anthropic", "google"]}


@app.post("/analyze")
async def analyze_leads(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
    x_ai_provider: Optional[str] = Header(default="openai"),
):
    """Analyse a list of raw inquiry strings and return structured lead intelligence."""
    body = await request.json()
    inquiries_raw: list = body.get("inquiries", [])

    # Validate
    inquiries = [s.strip() for s in inquiries_raw if isinstance(s, str) and s.strip()]
    if not inquiries:
        raise HTTPException(
            status_code=422,
            detail="At least one non-empty inquiry string is required.",
        )

    service = get_service(x_api_key, x_ai_provider)

    numbered = "\n\n".join(
        f"--- INQUIRY #{i + 1} ---\n{text}" for i, text in enumerate(inquiries)
    )

    try:
        raw = await service.get_completion(
            system_prompt=ANALYZE_SYSTEM_PROMPT,
            user_prompt=(
                f"Analyse these {len(inquiries)} real estate inquiries "
                f"and return a JSON object with key 'leads' containing an array:\n\n"
                f"{numbered}"
            ),
            temperature=0.2,
            json_mode=True
        )
        parsed = json.loads(raw)
        leads = parsed.get("leads", []) if isinstance(parsed, dict) else parsed

        for i, lead in enumerate(leads):
            lead["id"] = i + 1
            lead["raw_text"] = inquiries[i] if i < len(inquiries) else ""
            try:
                lead["priority_score"] = max(1, min(10, int(lead.get("priority_score", 5))))
            except (TypeError, ValueError):
                lead["priority_score"] = 5

        return {"leads": leads}

    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/generate-reply")
async def generate_reply(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
    x_ai_provider: Optional[str] = Header(default="openai"),
):
    """Generate a personalised reply email for a given lead."""
    body = await request.json()
    lead: dict = body.get("lead", {})
    if not lead:
        raise HTTPException(status_code=422, detail="'lead' object is required.")

    service = get_service(x_api_key, x_ai_provider)

    user_prompt = (
        f"Write a personalised reply to this real estate buyer inquiry.\n\n"
        f"Lead Profile:\n"
        f"- Name: {lead.get('name', 'Unknown')}\n"
        f"- Budget: {lead.get('budget', 'Not stated')}\n"
        f"- Urgency: {lead.get('urgency', 'Unknown')}\n"
        f"- Sentiment: {lead.get('sentiment', 'Neutral')}\n"
        f"- Summary: {lead.get('summary', '')}\n\n"
        f"Original Inquiry:\n{lead.get('raw_text', '')}"
    )

    try:
        reply = await service.get_completion(
            system_prompt=REPLY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.7
        )
        return {"reply": reply}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/chat")
async def chat(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
    x_ai_provider: Optional[str] = Header(default="openai"),
):
    """Chatbot endpoint for engaging medium/low-urgency leads."""
    body = await request.json()
    lead_context: str = body.get("lead_context", "")
    messages: list = body.get("messages", [])

    if not messages:
        raise HTTPException(status_code=422, detail="'messages' list cannot be empty.")

    service = get_service(x_api_key, x_ai_provider)
    system_msg = CHATBOT_SYSTEM_PROMPT.format(lead_context=lead_context)

    try:
        reply = await service.get_completion(
            system_prompt=system_msg,
            user_prompt="", # Combined into messages
            messages=messages,
            temperature=0.75
        )
        return {"reply": reply}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
