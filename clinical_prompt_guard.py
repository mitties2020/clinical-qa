"""Runtime guard for Vivid Medi clinical AI prompt formatting.

This keeps legacy server prompts from forcing universal safety-netting or
Markdown-style output while the main app prompt code remains otherwise stable.
"""

import re
from urllib.parse import urlparse

try:
    import requests
except Exception:  # pragma: no cover - requests is available in production
    requests = None

STANDARD_SECTIONS = (
    "Summary\n"
    "Assessment\n"
    "Diagnosis\n"
    "Investigations\n"
    "Treatment\n"
    "Monitoring\n"
    "Follow-up & Safety Netting\n"
    "Red Flags\n"
    "References\n"
)

RELEVANCE_FORMAT = (
    "OUTPUT FORMAT:\n"
    "Use clear section heading lines that fit the clinician-selected consult type and documented facts.\n"
    "Do not use Markdown, asterisks, bold markers, or decorative symbols.\n"
    "Include Monitoring, Follow-up, Safety Netting, Red Flags, and References only when clinically relevant.\n"
)

PROMPT_GUARD = (
    "Clinical documentation guard:\n"
    "Plain text only. Do not use Markdown, asterisks, bold markers, or decorative symbols.\n"
    "Choose headings that fit the clinician-selected consult type and documented facts.\n"
    "Optimise for Australian medical documentation standards: concise, clinically robust, defensible, and useful for continuity of care.\n"
    "Include safety-netting, red flags, monitoring, follow-up, and references only when clinically relevant to risk, medication/procedure changes, diagnostic uncertainty, or the consult type.\n"
    "Do not force safety-netting or red-flag sections into low-risk administrative, renewal, script, referral, or documentation-only notes unless clinically warranted.\n"
    "For DVA-related notes, write to an audit-ready DVA documentation standard without inventing accepted conditions or entitlement details."
)


def normalise_system_prompt(prompt):
    """Remove legacy mandatory section forcing from DeepSeek system prompts."""
    prompt = str(prompt or "")
    mandatory_block = "OUTPUT FORMAT (MANDATORY):\n" + STANDARD_SECTIONS
    prompt = prompt.replace(mandatory_block + "\n", RELEVANCE_FORMAT + "\n")
    prompt = prompt.replace(mandatory_block, RELEVANCE_FORMAT)
    prompt = prompt.replace(
        "Then output clinical sections:\n" + STANDARD_SECTIONS,
        "Then output clinically relevant sections for the selected task.\n"
        "Include Monitoring, Follow-up, Safety Netting, Red Flags, and References only when clinically relevant.\n",
    )
    prompt = prompt.replace("OUTPUT FORMAT (MANDATORY):", "OUTPUT FORMAT:")
    prompt = prompt.replace(
        "Plain text only. Registrar-level depth. Australian practice framing.",
        "Plain text only. Do not use Markdown, asterisks, or bold markers. Registrar-level depth. Australian practice framing.",
    )
    if "Do not force safety-netting or red-flag sections" not in prompt:
        prompt = prompt.rstrip() + "\n\n" + PROMPT_GUARD
    return prompt


def should_guard_url(url):
    try:
        host = urlparse(str(url)).netloc.lower()
    except Exception:
        return False
    return "deepseek" in host


def install_prompt_guard():
    if requests is None or getattr(requests.Session, "_vivid_prompt_guard_installed", False):
        return

    original_post = requests.Session.post

    def guarded_post(self, url, *args, **kwargs):
        if should_guard_url(url):
            payload = kwargs.get("json")
            messages = payload.get("messages") if isinstance(payload, dict) else None
            if isinstance(messages, list) and messages:
                system_message = messages[0]
                if isinstance(system_message, dict) and system_message.get("role") == "system":
                    system_message["content"] = normalise_system_prompt(system_message.get("content", ""))
        return original_post(self, url, *args, **kwargs)

    requests.Session.post = guarded_post
    requests.Session._vivid_prompt_guard_installed = True


install_prompt_guard()
