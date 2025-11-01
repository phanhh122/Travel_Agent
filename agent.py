import json
import os
import unicodedata
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque
import hashlib
import pickle
import traceback

import gradio as gr
import requests
import torch
import gc

# Vector search
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Clear GPU memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ==================== SIMPLE CACHE ====================

class SimpleCache:
    def __init__(self, cache_dir: str = './cache', ttl: int = 3600):
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_path(self, key: str) -> str:
        filename = hashlib.md5(key.encode()).hexdigest() + ".pkl"
        return os.path.join(self.cache_dir, filename)
    
    def get(self, key: str):
        path = self._get_path(key)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    value, timestamp = pickle.load(f)
                    if self.ttl is None or (datetime.now().timestamp() - timestamp < self.ttl):
                        return value
            except Exception:
                return None
        return None
    
    def set(self, key: str, value):
        try:
            with open(self._get_path(key), 'wb') as f:
                pickle.dump((value, datetime.now().timestamp()), f)
        except Exception:
            pass


# ==================== WEATHER SERVICE ====================

class WeatherService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.cache = SimpleCache('./cache/weather', ttl=1800)
    
    def get_weather(self, city: str) -> Dict:
        if not city:
            return {'location': city, 'info': 'No location provided'}
        cache_key = f"weather_{city.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        if not self.api_key:
            return {'location': city, 'info': 'N/A (need API key)'}
        
        try:
            response = requests.get(
                f"{self.base_url}/weather",
                params={'q': f'{city},VN', 'appid': self.api_key, 'units': 'metric', 'lang': 'en'},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            result = {
                'location': city,
                'temp': f"{data['main']['temp']:.1f}°C",
                'feels_like': f"{data['main'].get('feels_like', 0):.1f}°C" if data['main'].get('feels_like') else None,
                'desc': data['weather'][0]['description']
            }
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            # don't crash; return friendly fallback
            return {'location': city, 'info': 'Unable to fetch weather'}


# ==================== CONVERSATION MEMORY ====================

class ConversationMemory:
    def __init__(self, max_size: int = 10):
        self.history = deque(maxlen=max_size)
    
    def add(self, user: str, bot: str):
        self.history.append({'user': user, 'bot': bot})
    
    def get(self, last_n:int=5):
        return list(self.history)[-last_n:]
    
    def clear(self):
        self.history.clear()


# ==================== HELPERS ====================

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = s.replace('-', ' ').replace("’", "'")
    # collapse multiple spaces
    s = " ".join(s.split())
    return s


# ==================== DATA MANAGER ====================

class TravelDataManager:
    def __init__(self, data_path: str):
        print("Loading travel data...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.cache = SimpleCache('./cache/embeddings', ttl=None)
        cached = self.cache.get('embeddings')
        
        if cached:
            print("Loading cached embeddings...")
            self.corpus, self.metadata, self.embeddings = cached
        else:
            print("🔨 Creating embeddings (first time)...")
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self._build_index()
        
        # build normalized name index for fast exact lookup
        self._build_name_index()
        print("Data ready!")
    
    def _build_index(self):
        self.corpus = []
        self.metadata = []
        
        for item in self.data:
            # Build searchable text
            parts = []
            for k in ('name_vi','name_en','description_vi','description_en','best_time_vi','best_time_en'):
                if k in item and item[k]:
                    parts.append(str(item[k]))
            
            # foods and attractions
            foods_vi = ' '.join([f.get('name','') for f in item.get('foods_vi',[])])
            foods_en = ' '.join([f.get('name','') for f in item.get('foods_en',[])])
            attrs_vi = ' '.join([a.get('name','') for a in item.get('top_attractions_vi',[])])
            attrs_en = ' '.join([a.get('name','') for a in item.get('top_attractions_en',[])])
            
            text = " ".join(parts + [foods_vi, foods_en, attrs_vi, attrs_en])
            self.corpus.append(normalize_text(text))
            self.metadata.append(item)
        
        self.embeddings = self.embedder.encode(self.corpus, show_progress_bar=True)
        self.cache.set('embeddings', (self.corpus, self.metadata, self.embeddings))
    
    def _build_name_index(self):
        """Create map: normalized name -> metadata entry (first match)."""
        self.name_index = {}
        for item in self.metadata:
            nv = normalize_text(item.get('name_vi',''))
            ne = normalize_text(item.get('name_en',''))
            if nv:
                self.name_index[nv] = item
            if ne:
                self.name_index[ne] = item
            # also support dropping diacritics + no accents variants of name
            short_nv = nv.replace(' ', '')
            short_ne = ne.replace(' ', '')
            if short_nv:
                self.name_index[short_nv] = item
            if short_ne:
                self.name_index[short_ne] = item
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not hasattr(self, 'embedder'):
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        query_norm = normalize_text(query)
        query_emb = self.embedder.encode([query_norm])
        sims = cosine_similarity(query_emb, self.embeddings)[0]
        top_idx = np.argsort(sims)[-top_k:][::-1]
        
        return [{'score': float(sims[i]), 'data': self.metadata[i]} for i in top_idx]
    
    def find_by_name(self, name: str) -> Optional[Dict]:
        """Try multiple normalized variants to find exact data item."""
        if not name:
            return None
        n = normalize_text(name)
        # direct lookup
        if n in self.name_index:
            return self.name_index[n]
        # try token match in name_index keys
        for key, item in self.name_index.items():
            if n in key or key in n:
                return item
        tokens = n.split()
        for token in tokens:
            for key, item in self.name_index.items():
                if token in key:
                    return item
        return None


# ==================== SMART TRAVEL AGENT ====================

class SmartTravelAgent:
    """Rule-based agent - Fast & Accurate"""
    
    def __init__(self, data_path: str):
        print("Initializing Smart Travel Agent (Rule-based)...")
        try:
            self.data_manager = TravelDataManager(data_path)
        except Exception as e:
            print("Failed loading travel data:", e)
            traceback.print_exc()
            raise
        
        self.weather = WeatherService()
        self.memory = ConversationMemory()
        print("Agent ready!")
    
    def chat(self, message: str) -> str:
        """Main chat function - INSTANT response"""
        try:
            if not message or not message.strip():
                return "Hello! Ask me about traveling in Vietnam 🇻🇳"
            
            msg = message.strip()
            msg_lower = msg.lower()
            
            # Detect language
            is_english = self._is_english(msg)
            
            # Detect intent
            intent = self._detect_intent(msg_lower)
            
            # Weather intent first (user asked explicit weather)
            if intent == 'weather':
                # try to find location name from user text first
                # prefer exact data item name if possible
                item = self._find_location_item(msg)
                if item:
                    city_for_api = item.get('name_en') or item.get('name_vi')
                else:
                    # fallback: try to extract simple place token
                    city_for_api = self._extract_place_token(msg) or None
                
                if city_for_api:
                    w = self.weather.get_weather(city_for_api)
                    if 'temp' in w:
                        if is_english:
                            resp = f"🌤️ Weather in {w['location']}: {w['temp']}, {w['desc']}"
                        else:
                            resp = f"🌤️ Thời tiết tại {w['location']}: {w['temp']}, {w['desc']}"
                    else:
                        resp = w.get('info','Unable to fetch weather')
                    self.memory.add(message, resp)
                    return resp
                else:
                    return "Please tell me which city/province (e.g., 'Hanoi' / 'Hà Nội')."
            
            # Try exact location match using normalized names (best)
            item = self._find_location_item(msg)
            if item:
                response = self._build_smart_response(item, intent, is_english)
                self.memory.add(message, response)
                return response
            
            # Else fallback to semantic search (embedding)
            results = self.data_manager.search(msg, top_k=2)
            if not results or results[0]['score'] < 0.3:
                return self._fallback_response(intent, is_english)
            
            top = results[0]['data']
            response = self._build_smart_response(top, intent, is_english)
            self.memory.add(message, response)
            return response
            
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            traceback.print_exc()
            return "Sorry, I encountered an error. Please try again!"
    
    def _is_english(self, text: str) -> bool:
        """Simple language detection"""
        english_words = ['what', 'when', 'where', 'how', 'tell', 'about', 'best', 
                        'food', 'weather', 'visit', 'time', 'cost', 'price']
        text_lower = text.lower()
        english_count = sum(1 for word in english_words if word in text_lower)
        return english_count > 0
    
    def _detect_intent(self, msg: str) -> str:
        """Detect user intent"""
        if any(w in msg for w in ['weather', 'thời tiết', 'nhiệt độ', 'temperature']):
            return 'weather'
        elif any(w in msg for w in ['food', 'eat', 'dish', 'món ăn', 'ăn', 'đặc sản']):
            return 'food'
        elif any(w in msg for w in ['visit', 'attraction', 'see', 'tham quan', 'chơi', 'đi đâu']):
            return 'attractions'
        elif any(w in msg for w in ['when', 'time', 'season', 'khi nào', 'thời gian', 'tháng']):
            return 'timing'
        elif any(w in msg for w in ['cost', 'price', 'budget', 'chi phí', 'giá', 'tiền']):
            return 'budget'
        elif any(w in msg for w in ['recommend', 'suggest', 'gợi ý', 'tư vấn', 'nên']):
            return 'recommend'
        else:
            return 'general'
    
    def _extract_place_token(self, text: str) -> Optional[str]:
        """Try to extract a single token that looks like a place (fallback)."""
        tokens = [t for t in normalize_text(text).split() if len(t) > 2]
        if tokens:
            return tokens[-1].title()  # best-effort
        return None
    
    def _find_location_item(self, text: str) -> Optional[Dict]:
        """Try to find exact metadata entry by normalized name or known keywords."""
        # first check direct name_index
        candidate = self.data_manager.find_by_name(text)
        if candidate:
            return candidate
        # as additional aid: check keys mapping (commonly used name variants)
        # check words inside message
        tokens = normalize_text(text).split()
        # try long token combos (2-3 tokens)
        for L in (3,2,1):
            for i in range(len(tokens)-L+1):
                phrase = " ".join(tokens[i:i+L])
                found = self.data_manager.find_by_name(phrase)
                if found:
                    return found
        return None
    
    def _build_smart_response(self, data: Dict, intent: str, is_english: bool) -> str:
        """Build smart response based on intent and language"""
        name_vi = data.get('name_vi', 'Unknown')
        name_en = data.get('name_en', name_vi)
        
        try:
            if intent == 'food':
                if is_english:
                    foods = data.get('foods_en', [])[:3]
                    if not foods:
                        foods = data.get('foods_vi', [])[:3]
                    food_list = '\n• '.join([f"{f.get('name')}: {f.get('desc')} (~{f.get('avg_price','?')})" for f in foods])
                    where = foods[0].get('where_to_try','local market') if foods else ''
                    return f"🍜 Local Dishes in {name_en}:\n• {food_list}\n\n💡 Try at: {where}"
                else:
                    foods = data.get('foods_vi', [])[:3]
                    if not foods:
                        foods = data.get('foods_en', [])[:3]
                    food_list = '\n• '.join([f"{f.get('name')}: {f.get('desc')} (~{f.get('avg_price','?')})" for f in foods])
                    where = foods[0].get('where_to_try','địa điểm địa phương') if foods else ''
                    return f"🍜 Món ăn đặc sản tại {name_vi}:\n• {food_list}\n\n💡 Thử tại: {where}"
            
            elif intent == 'attractions':
                if is_english:
                    attrs = data.get('top_attractions_en', [])[:3] or data.get('top_attractions_vi',[])[:3]
                    attr_list = '\n• '.join([f"{a.get('name')}: {a.get('desc')} (Entry: {a.get('price','?')}, Duration: {a.get('duration','?')})" for a in attrs])
                    return f"📍 Top Attractions in {name_en}:\n• {attr_list}"
                else:
                    attrs = data.get('top_attractions_vi', [])[:3] or data.get('top_attractions_en',[])[:3]
                    attr_list = '\n• '.join([f"{a.get('name')}: {a.get('desc')} (Vé: {a.get('price','?')}, Thời gian: {a.get('duration','?')})" for a in attrs])
                    return f"📍 Địa điểm tham quan tại {name_vi}:\n• {attr_list}"
            
            elif intent == 'timing':
                if is_english:
                    return f"⏰ Best time to visit {name_en}: {data.get('best_time_en') or data.get('best_time_vi')}\n📅 Recommended stay: {data.get('days_en') or data.get('days_vi')}"
                else:
                    return f"⏰ Thời gian tốt nhất đi {name_vi}: {data.get('best_time_vi')}\n📅 Nên dành: {data.get('days_vi')}"
            
            elif intent == 'budget':
                if is_english:
                    try:
                        budget = data['suggested_itineraries_en'][0]['cost_estimate_en']
                    except Exception:
                        budget = data.get('suggested_itineraries_vi',[{}])[0].get('cost_estimate_vi','?')
                    return f"💰 Estimated cost for {name_en}: {budget}\n⏱️ Duration: {data.get('days_en') or data.get('days_vi')}"
                else:
                    try:
                        budget = data['suggested_itineraries_vi'][0]['cost_estimate_vi']
                    except Exception:
                        budget = data.get('suggested_itineraries_en',[{}])[0].get('cost_estimate_en','?')
                    return f"💰 Chi phí ước tính cho {name_vi}: {budget}\n⏱️ Thời gian: {data.get('days_vi')}"
            
            elif intent == 'recommend':
                if is_english:
                    tips = '\n• '.join(data.get('travel_tips_en',[])[:3] or data.get('travel_tips_vi',[])[:3])
                    return f"💡 Travel Tips for {name_en}:\n• {tips}\n\n⏰ Best time: {data.get('best_time_en') or data.get('best_time_vi')}"
                else:
                    tips = '\n• '.join(data.get('travel_tips_vi',[])[:3] or data.get('travel_tips_en',[])[:3])
                    return f"💡 Lời khuyên khi đi {name_vi}:\n• {tips}\n\n⏰ Thời gian tốt: {data.get('best_time_vi')}"
            
            else:  # general
                if is_english:
                    desc = data.get('description_en') or data.get('description_vi')
                    return f"📍 {name_en}\n\n{desc}\n\n⏰ Best time: {data.get('best_time_en') or data.get('best_time_vi')}\n📅 Recommended stay: {data.get('days_en') or data.get('days_vi')}"
                else:
                    desc = data.get('description_vi') or data.get('description_en')
                    return f"📍 {name_vi}\n\n{desc}\n\n⏰ Thời gian tốt nhất: {data.get('best_time_vi')}\n📅 Nên đi: {data.get('days_vi')}"
        except Exception as e:
            print("Error building response:", e)
            traceback.print_exc()
            return "Sorry, I couldn't build the answer right now."
    
    def _fallback_response(self, intent: str, is_english: bool) -> str:
        """Fallback when no match found"""
        if is_english:
            responses = {
                'food': "I can help you discover local dishes. Which city are you interested in?",
                'attractions': "I can recommend attractions. Which destination would you like?",
                'timing': "I can advise on best times. Which location are you planning?",
                'budget': "I can estimate trip cost. Which destination?",
            }
            return responses.get(intent,
                "🇻🇳 I can help you with:\n"
                "• Attractions & activities\n"
                "• Local food & specialties\n"
                "• Best travel times\n"
                "• Budget & costs\n"
                "• Weather\n\n"
                "Examples:\n• 'Tell me about Hanoi'\n• 'What to eat in Da Nang?'\n• 'Best time to visit Phu Quoc?'"
            )
        else:
            responses = {
                'food': "Bạn muốn biết món ăn đặc sản ở tỉnh/thành nào?",
                'attractions': "Bạn muốn gợi ý địa điểm tham quan ở đâu?",
                'timing': "Bạn muốn biết thời gian tốt nhất cho địa điểm nào?",
                'budget': "Bạn muốn ước tính chi phí ở điểm nào?",
            }
            return responses.get(intent,
                "🇻🇳 Tôi có thể giúp:\n"
                "• Địa điểm & hoạt động\n"
                "• Món ăn & đặc sản\n"
                "• Thời gian du lịch tốt nhất\n"
                "• Ngân sách & chi phí\n"
                "• Thời tiết\n\n"
                "Ví dụ: 'Giới thiệu về Hà Nội', 'Đà Nẵng có món gì ngon?'"
            )

