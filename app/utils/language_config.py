"""
language_config.py — 22 Official Indian Languages Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complete metadata for all 22 languages scheduled in the Eighth Schedule
of the Indian Constitution. Used across the ASR, TTS, and VLM pipelines.

This module is the linguistic foundation of Aura-NPU's "Linguistic Mirror"
feature — enabling real-time code-switching for 290 million learners
who study STEM in a non-native language.
"""

from __future__ import annotations

# ── Complete 22-Language Configuration Map ────────────────────────────────────
# Keys are BCP-47 language codes used across all Aura-NPU modules.

LANGUAGE_MAP: dict[str, dict] = {
    # ── Devanagari Script ──────────────────────────────────────────────────────
    "hi": {
        "name_en": "Hindi",
        "name_native": "हिन्दी",
        "script": "Devanagari",
        "iso_639_1": "hi",
        "iso_639_2": "hin",
        "rtl": False,
        "speakers_million": 600,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large",
        "tts_voice_id": "svara_hi_female_v2",
        "prompt_suffix": "हिंदी में उत्तर दें।",
        "states": ["Uttar Pradesh", "Bihar", "Madhya Pradesh", "Rajasthan", "Delhi"],
    },
    "mr": {
        "name_en": "Marathi",
        "name_native": "मराठी",
        "script": "Devanagari",
        "iso_639_1": "mr",
        "iso_639_2": "mar",
        "rtl": False,
        "speakers_million": 95,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large",
        "tts_voice_id": "svara_mr_female_v1",
        "prompt_suffix": "मराठीत उत्तर द्या।",
        "states": ["Maharashtra", "Goa"],
    },
    "sa": {
        "name_en": "Sanskrit",
        "name_native": "संस्कृत",
        "script": "Devanagari",
        "iso_639_1": "sa",
        "iso_639_2": "san",
        "rtl": False,
        "speakers_million": 0.025,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_sa_hybrid_rnnt_large",
        "tts_voice_id": "svara_sa_neutral_v1",
        "prompt_suffix": "संस्कृते उत्तरं ददातु।",
        "states": ["Classical — National"],
    },
    "kok": {
        "name_en": "Konkani",
        "name_native": "कोंकणी",
        "script": "Devanagari",
        "iso_639_1": "kok",
        "iso_639_2": "kok",
        "rtl": False,
        "speakers_million": 7.6,
        "ncert_available": False,
        "conformer_model": "ai4bharat/indicconformer_stt_kok_hybrid_rnnt_large",
        "tts_voice_id": "svara_kok_neutral_v1",
        "prompt_suffix": "कोंकणीत जवाप दी।",
        "states": ["Goa", "Karnataka", "Maharashtra"],
    },
    "brx": {
        "name_en": "Bodo",
        "name_native": "बड़ो",
        "script": "Devanagari",
        "iso_639_1": "brx",
        "iso_639_2": "brx",
        "rtl": False,
        "speakers_million": 1.5,
        "ncert_available": False,
        "conformer_model": "ai4bharat/indicconformer_stt_brx_hybrid_rnnt_large",
        "tts_voice_id": "svara_brx_neutral_v1",
        "prompt_suffix": "बड़ो भाषाय जोब आयो।",
        "states": ["Assam"],
    },
    "doi": {
        "name_en": "Dogri",
        "name_native": "डोगरी",
        "script": "Devanagari",
        "iso_639_1": "doi",
        "iso_639_2": "doi",
        "rtl": False,
        "speakers_million": 2.6,
        "ncert_available": False,
        "conformer_model": "ai4bharat/indicconformer_stt_doi_hybrid_rnnt_large",
        "tts_voice_id": "svara_doi_neutral_v1",
        "prompt_suffix": "डोगरी च जवाब देओ।",
        "states": ["Jammu & Kashmir"],
    },
    "mai": {
        "name_en": "Maithili",
        "name_native": "मैथिली",
        "script": "Devanagari",
        "iso_639_1": "mai",
        "iso_639_2": "mai",
        "rtl": False,
        "speakers_million": 35,
        "ncert_available": False,
        "conformer_model": "ai4bharat/indicconformer_stt_mai_hybrid_rnnt_large",
        "tts_voice_id": "svara_mai_female_v1",
        "prompt_suffix": "मैथिली में उत्तर दी।",
        "states": ["Bihar", "Jharkhand"],
    },
    # ── Dravidian Languages ─────────────────────────────────────────────────
    "ta": {
        "name_en": "Tamil",
        "name_native": "தமிழ்",
        "script": "Tamil",
        "iso_639_1": "ta",
        "iso_639_2": "tam",
        "rtl": False,
        "speakers_million": 80,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large",
        "tts_voice_id": "svara_ta_female_v2",
        "prompt_suffix": "தமிழில் பதில் அளிக்கவும்.",
        "states": ["Tamil Nadu", "Puducherry"],
    },
    "te": {
        "name_en": "Telugu",
        "name_native": "తెలుగు",
        "script": "Telugu",
        "iso_639_1": "te",
        "iso_639_2": "tel",
        "rtl": False,
        "speakers_million": 95,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_te_hybrid_rnnt_large",
        "tts_voice_id": "svara_te_female_v2",
        "prompt_suffix": "తెలుగులో సమాధానం ఇవ్వండి.",
        "states": ["Andhra Pradesh", "Telangana"],
    },
    "kn": {
        "name_en": "Kannada",
        "name_native": "ಕನ್ನಡ",
        "script": "Kannada",
        "iso_639_1": "kn",
        "iso_639_2": "kan",
        "rtl": False,
        "speakers_million": 60,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_kn_hybrid_rnnt_large",
        "tts_voice_id": "svara_kn_female_v1",
        "prompt_suffix": "ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
        "states": ["Karnataka"],
    },
    "ml": {
        "name_en": "Malayalam",
        "name_native": "മലയാളം",
        "script": "Malayalam",
        "iso_639_1": "ml",
        "iso_639_2": "mal",
        "rtl": False,
        "speakers_million": 38,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_ml_hybrid_rnnt_large",
        "tts_voice_id": "svara_ml_female_v1",
        "prompt_suffix": "മലയാളത്തിൽ മറുപടി നൽകുക.",
        "states": ["Kerala", "Lakshadweep"],
    },
    # ── Eastern Languages ───────────────────────────────────────────────────
    "bn": {
        "name_en": "Bengali",
        "name_native": "বাংলা",
        "script": "Bengali",
        "iso_639_1": "bn",
        "iso_639_2": "ben",
        "rtl": False,
        "speakers_million": 100,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_bn_hybrid_rnnt_large",
        "tts_voice_id": "svara_bn_female_v2",
        "prompt_suffix": "বাংলায় উত্তর দিন।",
        "states": ["West Bengal", "Tripura"],
    },
    "as": {
        "name_en": "Assamese",
        "name_native": "অসমীয়া",
        "script": "Bengali",
        "iso_639_1": "as",
        "iso_639_2": "asm",
        "rtl": False,
        "speakers_million": 15,
        "ncert_available": False,
        "conformer_model": "ai4bharat/indicconformer_stt_as_hybrid_rnnt_large",
        "tts_voice_id": "svara_as_female_v1",
        "prompt_suffix": "অসমীয়াত উত্তৰ দিয়ক।",
        "states": ["Assam"],
    },
    "or": {
        "name_en": "Odia",
        "name_native": "ଓଡ଼ିଆ",
        "script": "Odia",
        "iso_639_1": "or",
        "iso_639_2": "ori",
        "rtl": False,
        "speakers_million": 45,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_or_hybrid_rnnt_large",
        "tts_voice_id": "svara_or_female_v1",
        "prompt_suffix": "ଓଡ଼ିଆରେ ଉତ୍ତର ଦିଅ।",
        "states": ["Odisha"],
    },
    "mni": {
        "name_en": "Manipuri (Meitei)",
        "name_native": "মৈতৈলোন্",
        "script": "Meitei Mayek",
        "iso_639_1": "mni",
        "iso_639_2": "mni",
        "rtl": False,
        "speakers_million": 1.8,
        "ncert_available": False,
        "conformer_model": "ai4bharat/indicconformer_stt_mni_hybrid_rnnt_large",
        "tts_voice_id": "svara_mni_neutral_v1",
        "prompt_suffix": "মৈতৈলোন্ দা মীৎয়েং উৎলবিয়ু।",
        "states": ["Manipur"],
    },
    # ── Western Languages ────────────────────────────────────────────────────
    "gu": {
        "name_en": "Gujarati",
        "name_native": "ગુજરાતી",
        "script": "Gujarati",
        "iso_639_1": "gu",
        "iso_639_2": "guj",
        "rtl": False,
        "speakers_million": 60,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_gu_hybrid_rnnt_large",
        "tts_voice_id": "svara_gu_female_v1",
        "prompt_suffix": "ગુજરાતીમાં જવાબ આપો.",
        "states": ["Gujarat", "Dadra & Nagar Haveli"],
    },
    "pa": {
        "name_en": "Punjabi",
        "name_native": "ਪੰਜਾਬੀ",
        "script": "Gurmukhi",
        "iso_639_1": "pa",
        "iso_639_2": "pan",
        "rtl": False,
        "speakers_million": 33,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_pa_hybrid_rnnt_large",
        "tts_voice_id": "svara_pa_female_v1",
        "prompt_suffix": "ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ।",
        "states": ["Punjab", "Haryana", "Delhi"],
    },
    # ── Perso-Arabic Script ──────────────────────────────────────────────────
    "ur": {
        "name_en": "Urdu",
        "name_native": "اردو",
        "script": "Nastaliq (Perso-Arabic)",
        "iso_639_1": "ur",
        "iso_639_2": "urd",
        "rtl": True,  # Right-to-left script
        "speakers_million": 70,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_ur_hybrid_rnnt_large",
        "tts_voice_id": "svara_ur_female_v1",
        "prompt_suffix": "اردو میں جواب دیں۔",
        "states": ["Jammu & Kashmir", "Telangana", "Uttar Pradesh"],
    },
    "ks": {
        "name_en": "Kashmiri",
        "name_native": "كٲشُر",
        "script": "Perso-Arabic / Devanagari",
        "iso_639_1": "ks",
        "iso_639_2": "kas",
        "rtl": True,
        "speakers_million": 7,
        "ncert_available": False,
        "conformer_model": "ai4bharat/indicconformer_stt_ks_hybrid_rnnt_large",
        "tts_voice_id": "svara_ks_neutral_v1",
        "prompt_suffix": "كٲشُرِ زَبانَس مٕنز جواب دیو۔",
        "states": ["Jammu & Kashmir", "Ladakh"],
    },
    "sd": {
        "name_en": "Sindhi",
        "name_native": "سنڌي",
        "script": "Perso-Arabic / Devanagari",
        "iso_639_1": "sd",
        "iso_639_2": "snd",
        "rtl": True,
        "speakers_million": 25,
        "ncert_available": False,
        "conformer_model": "ai4bharat/indicconformer_stt_sd_hybrid_rnnt_large",
        "tts_voice_id": "svara_sd_neutral_v1",
        "prompt_suffix": "سنڌيءَ ۾ جواب ڏيو۔",
        "states": ["Gujarat", "Rajasthan"],
    },
    # ── Nepali / Other ──────────────────────────────────────────────────────
    "ne": {
        "name_en": "Nepali",
        "name_native": "नेपाली",
        "script": "Devanagari",
        "iso_639_1": "ne",
        "iso_639_2": "nep",
        "rtl": False,
        "speakers_million": 16,
        "ncert_available": False,
        "conformer_model": "ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large",
        "tts_voice_id": "svara_ne_female_v1",
        "prompt_suffix": "नेपालीमा जवाफ दिनुहोस्।",
        "states": ["Sikkim", "West Bengal (Darjeeling)"],
    },
    # ── English baseline ────────────────────────────────────────────────────
    "en": {
        "name_en": "English",
        "name_native": "English",
        "script": "Latin",
        "iso_639_1": "en",
        "iso_639_2": "eng",
        "rtl": False,
        "speakers_million": 125,
        "ncert_available": True,
        "conformer_model": "ai4bharat/indicconformer_stt_en_hybrid_rnnt_large",
        "tts_voice_id": "svara_en_female_v1",
        "prompt_suffix": "Please respond in clear, simple English.",
        "states": ["National — Official Language"],
    },
}


# ── Convenience Functions ─────────────────────────────────────────────────────

def get_language_names() -> dict[str, str]:
    """Return {code: 'Native Name (English Name)'} for UI dropdowns."""
    return {
        code: f"{info['name_native']} ({info['name_en']})"
        for code, info in LANGUAGE_MAP.items()
    }


def get_rtl_languages() -> list[str]:
    """Return list of RTL language codes (Urdu, Kashmiri, Sindhi)."""
    return [code for code, info in LANGUAGE_MAP.items() if info.get("rtl")]


def get_conformer_model_id(language_code: str) -> str:
    """Return the AI4Bharat IndicConformer model ID for a language."""
    return LANGUAGE_MAP.get(language_code, LANGUAGE_MAP["en"]).get(
        "conformer_model", "ai4bharat/indicconformer_stt_en_hybrid_rnnt_large"
    )


def get_prompt_suffix(language_code: str) -> str:
    """Return the language-appropriate prompt instruction suffix."""
    return LANGUAGE_MAP.get(language_code, LANGUAGE_MAP["en"]).get(
        "prompt_suffix", "Respond in English."
    )


def get_total_coverage() -> dict:
    """Return coverage statistics for the hackathon pitch."""
    total_speakers = sum(
        info["speakers_million"] for info in LANGUAGE_MAP.values() if info.get("speakers_million")
    )
    ncert_langs = sum(1 for info in LANGUAGE_MAP.values() if info.get("ncert_available"))
    return {
        "total_languages": len(LANGUAGE_MAP),
        "total_speakers_million": round(total_speakers, 1),
        "ncert_covered_languages": ncert_langs,
        "rtl_languages": len(get_rtl_languages()),
        "scheduled_languages_coverage": f"{len(LANGUAGE_MAP)}/22 (100%)",
    }
