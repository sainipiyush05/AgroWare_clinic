"""
=============================================================================
AgroWare — Disease Prediction with Remedies
=============================================================================
Complete system that detects diseases and provides treatment recommendations.

Usage:
    python predict_with_remedies.py --image path/to/leaf_image.jpg
    python predict_with_remedies.py --image path/to/leaf_image.jpg --remedies disease_remedies.json
    python predict_with_remedies.py --batch --images "folder/*.jpg" --output report.csv
=============================================================================
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Fix for Keras 3 model loading errors
import json
import argparse
import glob
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime


import sys
try:
    from tqdm import tqdm
except ImportError:
    # Create a simple fallback if tqdm not installed
    class tqdm:
        def __init__(self, total, desc="", unit=""):
            self.total = total
            self.n = 0
        def update(self, n=1):
            self.n += n
        def set_description(self, desc):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_PATH = "agroware_model.h5"
DEFAULT_TFLITE_PATH = "agroware_model.tflite"
DEFAULT_LABELS_PATH = "class_labels.json"
DEFAULT_REMEDIES_PATH = "disease_remedies_in.json"
IMG_SIZE = 224
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.3

# =============================================================================
# Remedy Database Functions (UPDATED FOR MULTILINGUAL SUPPORT)
# =============================================================================

class RemedyDatabase:
    """Load and query disease remedies database with multilingual support."""
    
    def __init__(self, remedies_path=DEFAULT_REMEDIES_PATH, language='en'):
        self.remedies_path = remedies_path
        self.language = language  # 'en', 'hi', or 'pa'
        self.remedies_db = self.load_remedies()
    
    def set_language(self, language):
        """Change the output language."""
        if language in ['en', 'hi', 'pa']:
            self.language = language
            print(f"🌐 Language set to: {language}")
        else:
            print(f"⚠️ Unsupported language: {language}. Using English.")
            self.language = 'en'
    
    def load_remedies(self):
        """Load remedies from JSON file."""
        if not os.path.exists(self.remedies_path):
            print(f"⚠️ Remedies file not found: {self.remedies_path}")
            print("   Creating template remedies file...")
            return self.create_template_remedies()
        
        with open(self.remedies_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_template_remedies(self):
        """Create a template remedies file if none exists."""
        template = {
            "Sample_Disease": {
                "crop": {"en": "Sample Crop", "hi": "नमूना फसल", "pa": "ਨਮੂਨਾ ਫ਼ਸਲ"},
                "disease": {"en": "Sample Disease", "hi": "नमूना रोग", "pa": "ਨਮੂਨਾ ਰੋਗ"},
                "symptoms": {
                    "en": ["Symptom 1", "Symptom 2"],
                    "hi": ["लक्षण 1", "लक्षण 2"],
                    "pa": ["ਲੱਛਣ 1", "ਲੱਛਣ 2"]
                },
                "chemical_control": {
                    "en": ["Chemical 1", "Chemical 2"],
                    "hi": ["रासायनिक 1", "रासायनिक 2"],
                    "pa": ["ਰਸਾਇਣ 1", "ਰਸਾਇਣ 2"]
                }
            }
        }
        
        with open(self.remedies_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Template remedies file created: {self.remedies_path}")
        return template
    
    def get_text_in_language(self, data_field):
        """Extract text in the selected language from a multilingual field."""
        if isinstance(data_field, dict):
            # If it's a dictionary with language keys
            if self.language in data_field:
                return data_field[self.language]
            elif 'en' in data_field:
                return data_field['en']  # Fallback to English
            else:
                # If no language keys, return as is
                return data_field
        else:
            # If it's not a dict (string or list), return as is
            return data_field
    
    def get_remedies(self, disease_class, confidence=None):
        """
        Get remedies for a disease class in the selected language.
        """
        # Handle healthy plants
        if "healthy" in disease_class.lower():
            return self.get_healthy_maintenance(disease_class)
        
        # Try exact match
        if disease_class in self.remedies_db:
            remedies = self.extract_remedies_in_language(self.remedies_db[disease_class])
            remedies['confidence'] = confidence
            return remedies
        
        # Try case-insensitive match
        for key in self.remedies_db:
            if key.lower() == disease_class.lower():
                remedies = self.extract_remedies_in_language(self.remedies_db[key])
                remedies['confidence'] = confidence
                return remedies
        
        # Try partial match
        disease_name = disease_class.split('_', 1)[-1] if '_' in disease_class else disease_class
        for key in self.remedies_db:
            if disease_name.lower() in key.lower():
                remedies = self.extract_remedies_in_language(self.remedies_db[key])
                remedies['confidence'] = confidence
                return remedies
        
        # If no match found, provide a generic fallback
        crop_name = disease_class.split('_')[0] if '_' in disease_class else "Unknown Crop"
        disease_display = ' '.join(disease_class.split('_')[1:]) if '_' in disease_class else disease_class
        
        generic_fallback = {
            "en": {
                "crop": crop_name,
                "disease": disease_display,
                "symptoms": [
                    "Symptoms specific to this disease are not currently in our database.", 
                    "Please consult a local agricultural expert for confirmation."
                ],
                "cultural_practices": [
                    "Remove and destroy infected plant parts immediately to prevent spread.", 
                    "Ensure proper drainage in the field to prevent waterlogging."
                ],
                "chemical_control": [
                    "Consult your local agriculture extension office or a trusted agro-chemical dealer for the specific recommended fungicide or pesticide."
                ],
                "prevention": [
                    "Practice crop rotation and field sanitation.", 
                    "Always use certified disease-free seeds or healthy planting material."
                ]
            },
            "hi": {
                "crop": crop_name,
                "disease": disease_display,
                "symptoms": [
                    "वर्तमान डेटाबेस में इस विशिष्ट रोग के लक्षण उपलब्ध नहीं हैं।", 
                    "कृपया पुष्टि के लिए स्थानीय कृषि विशेषज्ञ से परामर्श लें।"
                ],
                "cultural_practices": [
                    "फैलने से रोकने के लिए संक्रमित पौधों के हिस्सों को तुरंत हटाएं और नष्ट करें।", 
                    "जलभराव को रोकने के लिए खेत में उचित जल निकासी सुनिश्चित करें।"
                ],
                "chemical_control": [
                    "विशिष्ट अनुशंसित फफूंदनाशक या कीटनाशक के लिए अपने स्थानीय कृषि विस्तार कार्यालय या विश्वसनीय कृषि-रसायन डीलर से परामर्श लें।"
                ],
                "prevention": [
                    "फसल चक्र और खेत की स्वच्छता अपनाएं।", 
                    "हमेशा प्रमाणित रोग-मुक्त बीजों या स्वस्थ रोपण सामग्री का उपयोग करें।"
                ]
            },
            "pa": {
                "crop": crop_name,
                "disease": disease_display,
                "symptoms": [
                    "ਮੌਜੂਦਾ ਡੇਟਾਬੇਸ ਵਿੱਚ ਇਸ ਵਿਸ਼ੇਸ਼ ਰੋਗ ਦੇ ਲੱਛਣ ਉਪਲਬਧ ਨਹੀਂ ਹਨ।", 
                    "ਕਿਰਪਾ ਕਰਕੇ ਪੁਸ਼ਟੀ ਲਈ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਮਾਹਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।"
                ],
                "cultural_practices": [
                    "ਫੈਲਣ ਤੋਂ ਰੋਕਣ ਲਈ ਪ੍ਰਭਾਵਿਤ ਪੌਦੇ ਦੇ ਹਿੱਸਿਆਂ ਨੂੰ ਤੁਰੰਤ ਹਟਾਓ ਅਤੇ ਨਸ਼ਟ ਕਰੋ।", 
                    "ਪਾਣੀ ਭਰਾਅ ਨੂੰ ਰੋਕਣ ਲਈ ਖੇਤ ਵਿੱਚ ਉਚਿਤ ਪਾਣੀ ਨਿਕਾਸੀ ਯਕੀਨੀ ਬਣਾਓ।"
                ],
                "chemical_control": [
                    "ਵਿਸ਼ੇਸ਼ ਸਿਫਾਰਸ਼ ਕੀਤੇ ਫੰਗੀਸਾਈਡ ਜਾਂ ਕੀਟਨਾਸ਼ਕ ਲਈ ਆਪਣੇ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਵਿਸਥਾਰ ਦਫ਼ਤਰ ਜਾਂ ਭਰੋਸੇਯੋਗ ਖੇਤੀ-ਰਸਾਇਣ ਡੀਲਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।"
                ],
                "prevention": [
                    "ਫ਼ਸਲ ਚੱਕਰ ਅਤੇ ਖੇਤ ਦੀ ਸਫਾਈ ਅਪਣਾਓ।", 
                    "ਹਮੇਸ਼ਾ ਪ੍ਰਮਾਣਿਤ ਰੋਗ-ਮੁਕਤ ਬੀਜਾਂ ਜਾਂ ਸਿਹਤਮੰਦ ਬਿਜਾਈ ਸਮੱਗਰੀ ਦੀ ਵਰਤੋਂ ਕਰੋ।"
                ]
            }
        }
        
        fallback = generic_fallback[self.language].copy()
        fallback['confidence'] = confidence
        return fallback
    
    def extract_remedies_in_language(self, disease_data):
        """Extract all fields in the selected language."""
        remedies = {}
        
        # Map of fields to extract
        fields_to_extract = [
            'crop', 'disease', 'symptoms', 'weather_conditions', 'season',
            'regions', 'chemical_control', 'organic_remedies', 'prevention',
            'cultural_practices', 'severity_levels', 'time_to_recover',
            'yield_impact', 'emergency_contact', 'maintenance', 'preventive_sprays'
        ]
        
        for field in fields_to_extract:
            if field in disease_data:
                if field == 'severity_levels' and isinstance(disease_data[field], dict):
                    # Handle severity levels specially (nested structure)
                    remedies[field] = {}
                    for level, level_data in disease_data[field].items():
                        remedies[field][level] = self.get_text_in_language(level_data)
                elif field == 'regions' and isinstance(disease_data[field], dict):
                    # Regions might be language-specific lists
                    remedies[field] = self.get_text_in_language(disease_data[field])
                else:
                    remedies[field] = self.get_text_in_language(disease_data[field])
        
        return remedies
    
    def get_healthy_maintenance(self, crop_class):
        """Get maintenance tips for healthy plants in selected language."""
        crop = crop_class.split('_')[0]
        
        # Try to find crop-specific healthy tips
        healthy_key = f"{crop}_Healthy"
        if healthy_key in self.remedies_db:
            return self.extract_remedies_in_language(self.remedies_db[healthy_key])
        
        # Return generic healthy tips in selected language
        generic_tips = {
            "en": {
                "crop": crop,
                "disease": "Healthy",
                "symptoms": ["No disease symptoms detected"],
                "maintenance": [
                    "Continue regular monitoring",
                    "Maintain proper irrigation",
                    "Apply balanced fertilizers",
                    "Keep field weed-free",
                    "Practice crop rotation"
                ],
                "preventive_sprays": [
                    "Neem oil spray monthly",
                    "Apply Trichoderma viride @ 2.5kg/ha in soil",
                    "Foliar spray of micronutrients monthly"
                ]
            },
            "hi": {
                "crop": crop,
                "disease": "स्वस्थ",
                "symptoms": ["कोई रोग लक्षण नहीं पाए गए"],
                "maintenance": [
                    "नियमित निगरानी जारी रखें",
                    "उचित सिंचाई बनाए रखें",
                    "संतुलित उर्वरक लगाएं",
                    "खेत को खरपतवार मुक्त रखें",
                    "फसल चक्र अपनाएं"
                ],
                "preventive_sprays": [
                    "मासिक नीम तेल का छिड़काव",
                    "मिट्टी में ट्राइकोडर्मा विरिडे @ 2.5kg/हेक्टेयर लगाएं",
                    "मासिक सूक्ष्म पोषक तत्वों का पर्ण छिड़काव"
                ]
            },
            "pa": {
                "crop": crop,
                "disease": "ਸਿਹਤਮੰਦ",
                "symptoms": ["ਕੋਈ ਰੋਗ ਲੱਛਣ ਨਹੀਂ ਪਾਏ ਗਏ"],
                "maintenance": [
                    "ਨਿਯਮਤ ਨਿਗਰਾਨੀ ਜਾਰੀ ਰੱਖੋ",
                    "ਉਚਿਤ ਸਿੰਚਾਈ ਬਣਾਈ ਰੱਖੋ",
                    "ਸੰਤੁਲਿਤ ਖਾਦ ਲਗਾਓ",
                    "ਖੇਤ ਨੂੰ ਨਦੀਨ ਮੁਕਤ ਰੱਖੋ",
                    "ਫ਼ਸਲ ਚੱਕਰ ਅਪਣਾਓ"
                ],
                "preventive_sprays": [
                    "ਮਾਸਿਕ ਨਿੰਮ ਦੇ ਤੇਲ ਦਾ ਛਿੜਕਾਅ",
                    "ਮਿੱਟੀ ਵਿੱਚ ਟ੍ਰਾਈਕੋਡਰਮਾ ਵਿਰਾਈਡ @ 2.5kg/ਹੈਕਟੇਅਰ ਲਗਾਓ",
                    "ਮਾਸਿਕ ਸੂਖਮ ਪੌਸ਼ਟਿਕ ਤੱਤਾਂ ਦਾ ਫੋਲੀਅਰ ਛਿੜਕਾਅ"
                ]
            }
        }
        
        return generic_tips[self.language]
    
    def format_remedies_text(self, remedies):
        """Format remedies as readable text in the selected language."""
        if not remedies:
            return "No remedy information available for this disease."
        
        # Language-specific headers
        headers = {
            "en": {
                "title": "🌱 DISEASE MANAGEMENT RECOMMENDATIONS",
                "disease": "📋 Disease",
                "crop": "🌾 Crop",
                "confidence": "📊 Detection Confidence",
                "symptoms": "🔍 SYMPTOMS TO CONFIRM",
                "severity": "⚠️ SEVERITY GUIDE",
                "chemical": "🧪 CHEMICAL CONTROL",
                "organic": "🌿 ORGANIC/BIOLOGICAL CONTROL",
                "prevention": "🛡️ PREVENTION MEASURES",
                "cultural": "👨‍🌾 CULTURAL PRACTICES",
                "maintenance": "✅ MAINTENANCE TIPS",
                "recovery": "⏱️ Expected recovery time",
                "yield": "📉 Potential yield impact",
                "weather": "🌤️ Weather Conditions",
                "season": "📅 Season",
                "regions": "🗺️ Affected Regions",
                "emergency": "🚨 EMERGENCY",
                "footer": "⚠️ Always consult local agricultural extension for region-specific advice"
            },
            "hi": {
                "title": "🌱 रोग प्रबंधन सिफारिशें",
                "disease": "📋 रोग",
                "crop": "🌾 फसल",
                "confidence": "📊 पहचान विश्वसनीयता",
                "symptoms": "🔍 पुष्टि करने के लिए लक्षण",
                "severity": "⚠️ गंभीरता मार्गदर्शिका",
                "chemical": "🧪 रासायनिक नियंत्रण",
                "organic": "🌿 जैविक/जैविक नियंत्रण",
                "prevention": "🛡️ रोकथाम के उपाय",
                "cultural": "👨‍🌾 सांस्कृतिक पद्धतियां",
                "maintenance": "✅ रखरखाव युक्तियाँ",
                "recovery": "⏱️ अपेक्षित ठीक होने का समय",
                "yield": "📉 संभावित उपज प्रभाव",
                "weather": "🌤️ मौसम की स्थिति",
                "season": "📅 मौसम",
                "regions": "🗺️ प्रभावित क्षेत्र",
                "emergency": "🚨 आपातकालीन",
                "footer": "⚠️ क्षेत्र-विशिष्ट सलाह के लिए हमेशा स्थानीय कृषि विभाग से परामर्श लें"
            },
            "pa": {
                "title": "🌱 ਰੋਗ ਪ੍ਰਬੰਧਨ ਸਿਫ਼ਾਰਸ਼ਾਂ",
                "disease": "📋 ਰੋਗ",
                "crop": "🌾 ਫ਼ਸਲ",
                "confidence": "📊 ਪਛਾਣ ਭਰੋਸੇਯੋਗਤਾ",
                "symptoms": "🔍 ਪੁਸ਼ਟੀ ਕਰਨ ਲਈ ਲੱਛਣ",
                "severity": "⚠️ ਗੰਭੀਰਤਾ ਗਾਈਡ",
                "chemical": "🧪 ਰਸਾਇਣਕ ਨਿਯੰਤਰਣ",
                "organic": "🌿 ਜੈਵਿਕ/ਜੀਵਾਣੂ ਨਿਯੰਤਰਣ",
                "prevention": "🛡️ ਰੋਕਥਾਮ ਉਪਾਅ",
                "cultural": "👨‍🌾 ਸੱਭਿਆਚਾਰਕ ਅਭਿਆਸ",
                "maintenance": "✅ ਰੱਖ-ਰਖਾਅ ਸੁਝਾਅ",
                "recovery": "⏱️ ਅਨੁਮਾਨਿਤ ਠੀਕ ਹੋਣ ਦਾ ਸਮਾਂ",
                "yield": "📉 ਸੰਭਾਵੀ ਝਾੜ ਪ੍ਰਭਾਵ",
                "weather": "🌤️ ਮੌਸਮ ਦੀਆਂ ਸਥਿਤੀਆਂ",
                "season": "📅 ਰੁੱਤ",
                "regions": "🗺️ ਪ੍ਰਭਾਵਿਤ ਖੇਤਰ",
                "emergency": "🚨 ਐਮਰਜੈਂਸੀ",
                "footer": "⚠️ ਖੇਤਰ-ਵਿਸ਼ੇਸ਼ ਸਲਾਹ ਲਈ ਹਮੇਸ਼ਾ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਵਿਭਾਗ ਨਾਲ ਸਲਾਹ ਕਰੋ"
            }
        }
        
        h = headers[self.language]
        output = []
        output.append("\n" + "="*70)
        output.append(h["title"])
        output.append("="*70)
        
        # Basic info
        output.append(f"\n{h['disease']}: {remedies.get('disease', 'Unknown')}")
        output.append(f"{h['crop']}: {remedies.get('crop', 'Unknown')}")
        
        if 'confidence' in remedies and remedies['confidence']:
            output.append(f"{h['confidence']}: {remedies['confidence']:.1f}%")
        
        # Weather, Season, Regions (if available)
        if 'weather_conditions' in remedies:
            output.append(f"\n{h['weather']}: {remedies['weather_conditions']}")
        
        if 'season' in remedies:
            output.append(f"{h['season']}: {remedies['season']}")
        
        if 'regions' in remedies:
            if isinstance(remedies['regions'], list):
                output.append(f"{h['regions']}: {', '.join(remedies['regions'])}")
            else:
                output.append(f"{h['regions']}: {remedies['regions']}")
        
        # Symptoms
        if 'symptoms' in remedies:
            output.append(f"\n{h['symptoms']}:")
            for symptom in remedies['symptoms']:
                output.append(f"  • {symptom}")
        
        # Severity levels
        if 'severity_levels' in remedies:
            output.append(f"\n{h['severity']}:")
            for level, desc in remedies['severity_levels'].items():
                level_display = {
                    'low': '🟢 LOW' if self.language == 'en' else '🟢 कम' if self.language == 'hi' else '🟢 ਘੱਟ',
                    'medium': '🟡 MEDIUM' if self.language == 'en' else '🟡 मध्यम' if self.language == 'hi' else '🟡 ਦਰਮਿਆਨਾ',
                    'high': '🔴 HIGH' if self.language == 'en' else '🔴 उच्च' if self.language == 'hi' else '🔴 ਉੱਚ'
                }.get(level, level)
                output.append(f"  {level_display}: {desc}")
        
        # Chemical control
        if 'chemical_control' in remedies:
            output.append(f"\n{h['chemical']}:")
            for rec in remedies['chemical_control']:
                output.append(f"  • {rec}")
        
        # Organic remedies
        if 'organic_remedies' in remedies:
            output.append(f"\n{h['organic']}:")
            for rec in remedies['organic_remedies']:
                output.append(f"  • {rec}")
        
        # Prevention
        if 'prevention' in remedies:
            output.append(f"\n{h['prevention']}:")
            for rec in remedies['prevention']:
                output.append(f"  • {rec}")
        
        # Cultural practices
        if 'cultural_practices' in remedies:
            output.append(f"\n{h['cultural']}:")
            for rec in remedies['cultural_practices']:
                output.append(f"  • {rec}")
        
        # Maintenance (for healthy plants)
        if 'maintenance' in remedies:
            output.append(f"\n{h['maintenance']}:")
            for rec in remedies['maintenance']:
                output.append(f"  • {rec}")
        
        # Additional info
        if 'time_to_recover' in remedies:
            output.append(f"\n{h['recovery']}: {remedies['time_to_recover']}")
        
        if 'yield_impact' in remedies:
            output.append(f"{h['yield']}: {remedies['yield_impact']}")
        
        # Emergency contact
        if 'emergency_contact' in remedies:
            output.append(f"\n{h['emergency']}: {remedies['emergency_contact']}")
        
        output.append("\n" + "="*70)
        output.append(h["footer"])
        output.append("="*70)
        
        return "\n".join(output)
    
    def get_emergency_contact(self, disease_class):
        """Get emergency contact information from the remedies data."""
        if disease_class in self.remedies_db:
            data = self.remedies_db[disease_class]
            if 'emergency_contact' in data:
                return self.get_text_in_language(data['emergency_contact'])
        return None


# =============================================================================
# Prediction Functions (adapted from your existing code)
# =============================================================================

def load_and_preprocess_image(image_path, img_size=IMG_SIZE):
    """Load and preprocess image for MobileNetV2."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Could not open image {image_path}: {e}")
    
    if img.size[0] == 0 or img.size[1] == 0:
        raise ValueError(f"Image {image_path} has invalid dimensions")
    
    img = img.resize((img_size, img_size))
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_labels(labels_path):
    """Load class labels."""
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    if 'class_names' not in labels:
        if 'index_to_label' in labels:
            max_idx = max(int(k) for k in labels['index_to_label'].keys())
            class_names = [labels['index_to_label'][str(i)] for i in range(max_idx + 1)]
            labels['class_names'] = class_names
        else:
            raise ValueError("Labels file must contain 'class_names'")
    
    return labels


def predict_keras(image_path, model_path, labels_path):
    import tensorflow as tf

    print(f"📦 Loading Keras model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    labels = load_labels(labels_path)
    class_names = labels['class_names']

    img = load_and_preprocess_image(image_path)

    import time
    start = time.time()
    predictions = model.predict(img, verbose=0)[0]
    inference_time = (time.time() - start) * 1000
    
    num_classes = len(predictions)
    if len(class_names) < num_classes:
        print(f"⚠️ Warning: Model outputs {num_classes} classes but labels file has {len(class_names)}. Pad with defaults.")
        for i in range(len(class_names), num_classes):
            class_names.append(f"Class_{i}")

    return predictions, class_names, inference_time


def predict_tflite(image_path, tflite_path, labels_path):
    """Predict using TFLite model."""
    import tensorflow as tf
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    expected_height = input_shape[1]
    
    labels = load_labels(labels_path)
    class_names = labels['class_names']
    
    img = load_and_preprocess_image(image_path, img_size=expected_height)
    
    input_dtype = input_details[0]['dtype']
    if input_dtype != img.dtype:
        img = img.astype(input_dtype)
    
    import time
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    inference_time = (time.time() - start) * 1000
    
    return predictions.astype(np.float32), class_names, inference_time


def predict_with_remedies(image_path, model_path=None, tflite_path=None, 
                          labels_path=DEFAULT_LABELS_PATH, 
                          remedies_path=DEFAULT_REMEDIES_PATH,
                          language='en', crop_filter=None,
                          top_k=TOP_K, threshold=CONFIDENCE_THRESHOLD):
    """
    Complete prediction with remedies in specified language.
    """
    # Initialize remedy database with language
    remedy_db = RemedyDatabase(remedies_path, language)
    
    # Auto-detect model
    if not model_path and not tflite_path:
        potential_models = [
            "agroware_model.h5", "agroware_model.keras", 
            "final_model.keras", "best_final.keras",
            DEFAULT_TFLITE_PATH, DEFAULT_MODEL_PATH
        ]
        
        for p in potential_models:
            if os.path.exists(p):
                if p.endswith('.tflite'):
                    tflite_path = p
                else:
                    model_path = p
                break
        
        if not model_path and not tflite_path:
            raise FileNotFoundError("No model found. Place 'agroware_model.keras' in the root directory.")
    
    # Run prediction
    if tflite_path and os.path.exists(tflite_path):
        predictions, class_names, inference_time = predict_tflite(
            image_path, tflite_path, labels_path
        )
    elif model_path and os.path.exists(model_path):
        predictions, class_names, inference_time = predict_keras(
            image_path, model_path, labels_path
        )
    else:
        raise FileNotFoundError(f"Model not found")
    
    # Format predictions
    predictions_pct = predictions * 100
    top_indices = np.argsort(predictions)[::-1]
    
    results = []
    for idx in top_indices:
        class_label = class_names[idx]
        
        if crop_filter and crop_filter.lower() not in class_label.lower():
            continue
            
        confidence = float(predictions_pct[idx])
        
        # Get remedies for this disease
        remedies = remedy_db.get_remedies(class_label, confidence)
        emergency = remedy_db.get_emergency_contact(class_label)
        
        result = {
            'class_index': int(idx),
            'class_label': class_label,
            'crop': class_label.split('_')[0],
            'disease': '_'.join(class_label.split('_')[1:]),
            'disease_display': ' '.join(class_label.split('_')[1:]),
            'confidence': float(confidence),
            'confidence_decimal': float(predictions[idx]),
            'is_healthy': 'healthy' in class_label.lower(),
            'remedies': remedies,
            'emergency': emergency
        }
        results.append(result)
        
        if len(results) >= top_k:
            break
    
    if not results and crop_filter:
        print(f"⚠️ No matching predictions found for crop '{crop_filter}'.")
        
    # Add metadata
    metadata = {
        'image': image_path,
        'timestamp': datetime.now().isoformat(),
        'inference_time_ms': inference_time,
        'predictions': results
    }
    
    return metadata


def generate_report(results_list, output_format='text', language='en'):
    """Generate a report from multiple predictions in specified language."""
    
    # Language-specific column headers for CSV
    csv_headers = {
        "en": ['Image', 'Crop', 'Disease', 'Confidence', 'Status', 'Treatment'],
        "hi": ['छवि', 'फसल', 'रोग', 'विश्वसनीयता', 'स्थिति', 'उपचार'],
        "pa": ['ਤਸਵੀਰ', 'ਫ਼ਸਲ', 'ਰੋਗ', 'ਭਰੋਸੇਯੋਗਤਾ', 'ਸਥਿਤੀ', 'ਇਲਾਜ']
    }
    
    if output_format == 'csv':
        df = pd.DataFrame([
            {
                csv_headers[language][0]: os.path.basename(r['image']),
                csv_headers[language][1]: r['predictions'][0]['crop'],
                csv_headers[language][2]: r['predictions'][0]['disease_display'],
                csv_headers[language][3]: f"{r['predictions'][0]['confidence']:.1f}%",
                csv_headers[language][4]: '✅ स्वस्थ' if r['predictions'][0]['is_healthy'] and language=='hi' else '✅ ਸਿਹਤਮੰਦ' if language=='pa' else '✅ Healthy' if r['predictions'][0]['is_healthy'] else '⚠️ रोगी' if language=='hi' else '⚠️ ਰੋਗੀ' if language=='pa' else '⚠️ Diseased',
                csv_headers[language][5]: r['predictions'][0]['remedies'].get('chemical_control', ['N/A'])[0][:50] if r['predictions'][0]['remedies'] else 'N/A'
            }
            for r in results_list if r['success']
        ])
        return df
    return None

# =============================================================================
# CLI Interface (UPDATED WITH MULTILINGUAL SUPPORT)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AgroWare — Plant Disease Prediction with Remedies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # English output (default)
  python predict_with_remedies.py --image wheat.jpg
  
  # Hindi output
  python predict_with_remedies.py --image wheat.jpg --lang hi
  
  # Punjabi output with batch processing
  python predict_with_remedies.py --batch --images "field/*.jpg" --lang pa --output report_punjabi.csv
  
  # JSON output in Hindi
  python predict_with_remedies.py --image rice.jpg --lang hi --json
  
  # Specify custom remedies file
  python predict_with_remedies.py --image potato.jpg --remedies disease_remedies_in.json --lang pa
  
  # Low confidence threshold and more predictions
  python predict_with_remedies.py --image tomato.jpg --threshold 0.2 --top-k 5 --lang hi
        """
    )
    
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--image', help='Path to single leaf image')
    input_group.add_argument('--crop', default=None, help='Filter predictions by crop name (e.g., Tomato, Wheat)')
    input_group.add_argument('--batch', action='store_true', help='Batch processing mode')
    input_group.add_argument('--images', help='Glob pattern for batch (e.g., "folder/*.jpg" or "field/*.png")')
    
    # MODEL OPTIONS
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--model', default=None, help='Path to Keras model (.h5 or .keras)')
    model_group.add_argument('--tflite', default=None, help='Path to TFLite model (.tflite)')
    model_group.add_argument('--labels', default=DEFAULT_LABELS_PATH, help='Path to class labels JSON file')
    model_group.add_argument('--remedies', default=DEFAULT_REMEDIES_PATH, help='Path to remedies database JSON file')
    
    # LANGUAGE OPTIONS
    lang_group = parser.add_argument_group('Language Options')
    lang_group.add_argument('--lang', '--language', choices=['en', 'hi', 'pa'], default='en', help='Output language: en (English), hi (Hindi), pa (Punjabi)')
    lang_group.add_argument('--list-languages', action='store_true', help='List available languages and exit')
    
    # OUTPUT OPTIONS
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--top-k', type=int, default=TOP_K, help=f'Number of top predictions to show (default: {TOP_K})')
    output_group.add_argument('--threshold', type=float, default=CONFIDENCE_THRESHOLD, help=f'Confidence threshold 0-1 (default: {CONFIDENCE_THRESHOLD})')
    output_group.add_argument('--json', action='store_true', help='Output as JSON (overrides --format)')
    output_group.add_argument('--format', choices=['text', 'csv', 'json'], default='text', help='Output format (default: text)')
    output_group.add_argument('--output', help='Save output to file (instead of printing)')
    output_group.add_argument('--no-remedies', action='store_true', help='Skip remedies (faster, prediction only)')
    output_group.add_argument('--verbose', action='store_true', help='Show detailed output including all predictions')
    
    args = parser.parse_args()
    
    if args.list_languages:
        print("\n🌐 Available Languages:")
        print("   en  - English (ਅੰਗਰੇਜ਼ੀ)")
        print("   hi  - Hindi (हिन्दी)")
        print("   pa  - Punjabi (ਪੰਜਾਬੀ)")
        return 0
    
    if not args.image and not args.batch:
        parser.error("❌ Either --image or --batch is required")
    
    if args.batch and not args.images:
        parser.error("❌ --batch requires --images pattern (e.g., --images 'folder/*.jpg')")
    
    greetings = {
        "en": "🌐 Language: English",
        "hi": "🌐 भाषा: हिन्दी",
        "pa": "🌐 ਭਾਸ਼ਾ: ਪੰਜਾਬੀ"
    }
    print(greetings[args.lang])
    
    if not args.no_remedies:
        try:
            remedy_db = RemedyDatabase(args.remedies, args.lang)
            print(f"📚 Remedies database loaded: {os.path.basename(args.remedies)}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load remedies database: {e}")
            args.no_remedies = True
    
    results_list = []
    successful = 0
    failed = 0
    
    if args.image:
        print(f"\n🔍 Processing: {os.path.basename(args.image)}")
        try:
            result = predict_with_remedies(
                image_path=args.image,
                model_path=args.model,
                tflite_path=args.tflite,
                labels_path=args.labels,
                remedies_path=args.remedies if not args.no_remedies else None,
                language=args.lang,
                crop_filter=args.crop,
                top_k=args.top_k,
                threshold=args.threshold
            )
            result['success'] = True
            results_list.append(result)
            successful += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results_list.append({'image': args.image, 'success': False, 'error': str(e)})
            failed += 1
    
    elif args.batch:
        image_paths = glob.glob(args.images)
        if not image_paths:
            print(f"❌ No images found matching pattern: {args.images}")
            return 1
        image_paths.sort()
        print(f"\n📦 Found {len(image_paths)} images")
        
        from tqdm import tqdm
        with tqdm(total=len(image_paths), desc="Progress", unit="img") as pbar:
            for img_path in image_paths:
                try:
                    result = predict_with_remedies(
                        image_path=img_path,
                        model_path=args.model,
                        tflite_path=args.tflite,
                        labels_path=args.labels,
                        remedies_path=args.remedies if not args.no_remedies else None,
                        language=args.lang,
                        crop_filter=args.crop,
                        top_k=args.top_k,
                        threshold=args.threshold
                    )
                    result['success'] = True
                    results_list.append(result)
                    successful += 1
                except Exception as e:
                    results_list.append({'image': img_path, 'success': False, 'error': str(e)})
                    failed += 1
                pbar.update(1)
    
    output_str = ""
    if args.format == 'json' or args.json:
        output_data = results_list if args.batch else (results_list[0] if results_list else {})
        output_str = json.dumps(output_data, indent=2, default=str, ensure_ascii=False)
    elif args.format == 'csv' and args.batch:
        df = generate_report(results_list, 'csv', args.lang)
        output_str = df.to_csv(index=False, encoding='utf-8-sig')
    else:
        if args.batch:
            output_str = generate_report(results_list, 'text', args.lang)
        else:
            if results_list and results_list[0]['success']:
                result = results_list[0]
                headers = {
                    "en": {"title": "🌱 AGROWARE DISEASE PREDICTION RESULT", "image": "📷 Image", "time": "🕒 Time", "inference": "⚡ Inference"},
                    "hi": {"title": "🌱 एग्रोवेयर रोग भविष्यवाणी परिणाम", "image": "📷 छवि", "time": "🕒 समय", "inference": "⚡ अनुमान समय"},
                    "pa": {"title": "🌱 ਐਗਰੋਵੇਅਰ ਰੋਗ ਭਵਿੱਖਬਾਣੀ ਨਤੀਜਾ", "image": "📷 ਤਸਵੀਰ", "time": "🕒 ਸਮਾਂ", "inference": "⚡ ਅਨੁਮਾਨ ਸਮਾਂ"}
                }
                h = headers[args.lang]
                output_lines = ["\n" + "="*70, h["title"], "="*70]
                output_lines.append(f"{h['image']}: {os.path.basename(result['image'])}")
                output_lines.append(f"{h['time']}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                output_lines.append(f"{h['inference']}: {result['inference_time_ms']:.1f}ms")
                
                if result['predictions']:
                    top = result['predictions'][0]
                    status = ("✅ स्वस्थ" if args.lang == 'hi' else "✅ ਸਿਹਤਮੰਦ" if args.lang == 'pa' else "✅ HEALTHY") if top['is_healthy'] else \
                             ("⚠️ रोगी" if args.lang == 'hi' else "⚠️ ਰੋਗੀ" if args.lang == 'pa' else "⚠️ DISEASED")
                    output_lines.append(f"\n📊 Top Prediction: {top['class_label']}")
                    output_lines.append(f"   Confidence: {top['confidence']:.1f}%")
                    output_lines.append(f"   Status: {status}")
                    
                    if args.verbose:
                        output_lines.append(f"\n📋 All Top Predictions:")
                        for p in result['predictions']:
                            output_lines.append(f"   • {p['class_label']}: {p['confidence']:.1f}%")
                
                output_str = "\n".join(output_lines)
                if not args.no_remedies and result['predictions'] and result['predictions'][0].get('remedies'):
                    output_str += remedy_db.format_remedies_text(result['predictions'][0]['remedies'])
            else:
                output_str = "❌ No successful predictions"
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8-sig') as f:
            f.write(output_str)
        print(f"\n✅ Report saved to: {args.output}")
    else:
        print(output_str)
    
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())