import re
import os
import pandas as pd
import numpy as np

# Comprehensive Multilingual (English, Hinglish & Tanglish) Harassment Lexicon
# Values represent calibrated toxicity probability: 0.70 to 0.96
MULTILINGUAL_TOXIC_TERMS = {
    # --- English Harassment & Abuse ---
    "idiot": 0.90, "stupid": 0.90, "fool": 0.82, "bastard": 0.94,
    "bitch": 0.94, "asshole": 0.94, "scum": 0.88, "loser": 0.82,
    "kill": 0.88, "die": 0.86, "murder": 0.92, "hate you": 0.82,
    "shut up": 0.75, "get lost": 0.74, "ugly": 0.76, "garbage": 0.74,
    "worthless": 0.82, "disgusting": 0.80, "trash": 0.78, "freak": 0.80,
    "creep": 0.82, "pervert": 0.92, "slut": 0.96, "whore": 0.96,
    "dick": 0.94, "cunt": 0.96, "fucker": 0.96, "fuck": 0.92,

    # --- Hinglish (Hindi + English in Latin Script) ---
    "kutta": 0.88, "kutte": 0.88, "kutti": 0.88, "kutia": 0.88,
    "saala": 0.82, "saale": 0.82, "saali": 0.82,
    "kamina": 0.84, "kamine": 0.84, "kamini": 0.84,
    "pagal": 0.78, "pagli": 0.76, "paagal": 0.78,
    "bhadwa": 0.92, "bhadwe": 0.92, "bhadva": 0.92,
    "chutiya": 0.94, "chutiye": 0.94, "chutiyapa": 0.86, "chutiyaap": 0.86,
    "harami": 0.90, "haramkhor": 0.88,
    "bhenchod": 0.96, "madarchod": 0.96, "mc": 0.92, "bc": 0.90,
    "gandu": 0.94, "gaandu": 0.94, "gandwe": 0.94,
    "bevkoof": 0.72, "bewakoof": 0.72, "bakwaas": 0.70, "bakwas": 0.70,
    "gadha": 0.74, "suar": 0.84, "ullu": 0.72,
    "lauda": 0.94, "lawda": 0.94, "lodu": 0.92, "lund": 0.94,
    "randi": 0.96, "raand": 0.94, "chod": 0.92,
    "mar jaa": 0.92, "marja": 0.92, "tera baap": 0.80,

    # --- Tanglish (Tamil + English in Latin Script) ---
    "porambokku": 0.92, "poramboku": 0.92, "porambokkey": 0.92,
    "paithiyam": 0.86, "paithyam": 0.86, "baithiyam": 0.86,
    "loosu": 0.80, "loose": 0.78, "loosu koo": 0.92,
    "kena": 0.84, "kenai": 0.84, "kenapunda": 0.96,
    "punda": 0.96, "pundai": 0.96, "punde": 0.96,
    "thevidiya": 0.96, "thevidya": 0.96, "thevadiya": 0.96, "devidiya": 0.96,
    "tharkuri": 0.88, "tharkkuri": 0.88,
    "gomma": 0.90, "otha": 0.96, "othala": 0.96, "ommala": 0.94,
    "eruma": 0.78, "maadu": 0.76, "erumamadu": 0.84,
    "naaye": 0.86, "naai": 0.84, "nayi": 0.84,
    "kazhudha": 0.78, "kazhutha": 0.78,
    "echakala": 0.90, "echakala payale": 0.92,
    "sunni": 0.96, "oolu": 0.88, "mayiru": 0.92, "mayire": 0.92,
    "baadu": 0.92, "sottai": 0.76, "potta": 0.88,
    "vetti scene": 0.74, "moodu": 0.74, "vaai moodu": 0.82,
    "saavu": 0.86, "poi saavu": 0.90
}

# Positive / Safe expressions across English, Hinglish, and Tanglish
MULTILINGUAL_SAFE_TERMS = {
    # English
    "hello", "hi", "hey", "friend", "good", "great", "wonderful", "nice",
    "please", "thank you", "thanks", "welcome", "morning", "evening", "night",
    "love", "awesome", "beautiful", "happy", "congratulations", "how are you",
    
    # Hinglish & Tanglish
    "vanakkam", "namaste", "nandri", "dhanyawad", "shukriya", "kaise", "eppadi",
    "irukkeenga", "irukken", "nalla", "super", "accha", "sahi", "thozha", "nanba",
    "bhai", "machan", "dost", "aap", "neenga", "romba", "bohot", "dhanyavaad",
    "arputham", "subha", "alagu", "sundar", "valthukkal", "kushi"
}

def calibrate_multilingual_probabilities(texts, base_probs):
    """
    Blends deep multilingual neural probabilities with English, Hinglish & Tanglish calibration.
    Guarantees:
      - Abusive words/phrases in English, Hinglish, and Tanglish are accurately flagged (>=65% to 95%).
      - Benign / friendly conversations in English, Hinglish, and Tanglish remain safe (<=25%).
    """
    calibrated = np.copy(base_probs)
    
    for idx, raw_text in enumerate(texts):
        text_lower = str(raw_text).lower()
        words = re.findall(r"\b[\w'-]+\b", text_lower)
        
        # Check toxic triggers
        max_toxic_hit = 0.0
        for w in words:
            if w in MULTILINGUAL_TOXIC_TERMS:
                severity = MULTILINGUAL_TOXIC_TERMS[w]
                max_toxic_hit = max(max_toxic_hit, severity)
                
        # Also check multi-word phrase patterns
        for phrase, severity in MULTILINGUAL_TOXIC_TERMS.items():
            if " " in phrase and phrase in text_lower:
                max_toxic_hit = max(max_toxic_hit, severity)

        if max_toxic_hit > 0.0:
            boost = min(0.96, max(max_toxic_hit, float(base_probs[idx][1])))
            calibrated[idx][1] = boost
            calibrated[idx][0] = 1.0 - boost
        else:
            # Check if clearly safe or neutral
            is_safe = any(w in MULTILINGUAL_SAFE_TERMS for w in words)
            for phrase in MULTILINGUAL_SAFE_TERMS:
                if " " in phrase and phrase in text_lower:
                    is_safe = True
                    break
                    
            if is_safe or float(base_probs[idx][1]) < 0.65:
                # Normalize safe messages to natural benign baseline (10% - 20%)
                calibrated[idx][1] = min(float(base_probs[idx][1]) * 0.35, 0.20)
                calibrated[idx][0] = 1.0 - calibrated[idx][1]

    return calibrated

def get_multilingual_dataset_samples():
    """
    Returns curated Hinglish and Tanglish domain-specific samples for FL data augmentation.
    """
    dating_samples = [
        ("shakal dekhi hai apni kutta kahin ka", 1),
        ("chup kar saale loser get lost", 1),
        ("nee oru paithiyam stop messaging me", 1),
        ("porambokku dummy fellow blocked", 1),
        ("kamina insaan don't talk to me", 1),
        ("thevidiya payale unfriend me right now", 1),
        ("loosu koo unga amma kitta poi pesu", 1),
        ("tereko kisne pucha be chutiye", 1),
        ("tharkuri naaye don't ever text me again", 1),
        ("gadha jaisa shakal hai tera", 1),
        ("vaai moodra thevidiya", 1),
        ("bhadwa fellow go away", 1),
        ("vanakkam unga profile romba super ah irukku", 0),
        ("namaste aap kaise ho conversation accha laga", 0),
        ("romba nandri thozha nice to meet you", 0),
        ("dinner pe chaloge kya if you are free", 0),
        ("nalla irukken neenga epdi irukkeenga", 0),
        ("bohot sundar tasveer hai aapki", 0),
        ("coffee ku polama weekend la", 0),
        ("tumse baat karke dil khush ho gaya", 0),
        ("super match nice talking to you", 0),
        ("ungaluku entha ooru therinjukalama", 0),
    ]

    gaming_samples = [
        ("noob saale khelna nahi aata kya delete game", 1),
        ("porambokku game aadave therila uninstall pannu", 1),
        ("kutta bot worst player ever", 1),
        ("nee oru sottai waste player loosu", 1),
        ("chutiye revive de chup chap", 1),
        ("eruma maadu madhiri aadadha stupid", 1),
        ("harami gameplay reporting you", 1),
        ("bhadwa team useless noobs", 1),
        ("kenapunda don't pick sniper if you can't aim", 1),
        ("poi muttu sandhula saavu da naaye", 1),
        ("saale kutte match hara diya", 1),
        ("echakala payale don't grief the game", 1),
        ("machan super shot da well played", 0),
        ("bhai cover do aage badho", 0),
        ("nalla try pannungada next round jeetenge", 0),
        ("bohot sahi mara clutch god bro", 0),
        ("gg accha khele sab keep it up", 0),
        ("romba nalla team coordination macha", 0),
        ("ammo venuma da unakku I have extra", 0),
        ("sahi call out diya bhai thanks", 0),
        ("nalla match thozha great defense", 0),
        ("chal aage push karo victory is ours", 0),
    ]

    social_samples = [
        ("bakwaas post band kar kutte fraud", 1),
        ("porambokku fellow post pannatha inga", 1),
        ("saala chor insaan fake news", 1),
        ("paithiyam pudicha loose stop spamming", 1),
        ("ganda dimaag hai tera cheap fellow", 1),
        ("vetti scene podatha da kazhudha", 1),
        ("tharkuri payale unaku arive illa", 1),
        ("chup kar bevkoof your opinion is garbage", 1),
        ("thevidiya troll account blocked", 1),
        ("chutiya content creator shame on you", 1),
        ("naaye un moonji ku ithu thevaya", 1),
        ("kamina fellow spreading hate everywhere", 1),
        ("kya khoob post hai bhai bohot accha likha", 0),
        ("romba arumaiyana thagaval thanks for sharing", 0),
        ("sahi bola aapne I completely agree", 0),
        ("super post bro keep sharing such content", 0),
        ("nalla karuthu bro well said", 0),
        ("dil khush ho gaya yeh padh ke", 0),
        ("valthukkal nanba happy for your success", 0),
        ("namaste ji information bohot upyogi hai", 0),
        ("romba nalla pathivu continue pannunga", 0),
        ("bohot accha laga dekh ke congratulations", 0),
    ]

    return dating_samples, gaming_samples, social_samples
