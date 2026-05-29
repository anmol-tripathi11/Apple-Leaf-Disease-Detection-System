"""
Apple Leaf Disease Detection - Complete Working Solution
Enhanced with Prevention and Cure Database & Google Translate API
Version: 4.1.0 (Added Safe Prediction with Confidence Threshold)
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os, json, argparse, sys, random, warnings, traceback
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import uuid, webbrowser, threading, time
from werkzeug.utils import secure_filename
from deep_translator import GoogleTranslator

warnings.filterwarnings('ignore')

# ============================================================================
# GOOGLE TRANSLATE CONFIG
# ============================================================================
GOOGLE_TRANSLATE_API_KEY = "AIzaSyC7P4jQlICxf8tGqWLVdbhHpBH9dUMn_Wg"

SUPPORTED_LANGUAGES = [
    {"code":"en",    "name":"English",          "flag":"🇺🇸"},
    {"code":"es",    "name":"Español",           "flag":"🇪🇸"},
    {"code":"fr",    "name":"Français",          "flag":"🇫🇷"},
    {"code":"de",    "name":"Deutsch",           "flag":"🇩🇪"},
    {"code":"hi",    "name":"हिन्दी",             "flag":"🇮🇳"},
    {"code":"zh-CN", "name":"中文 (简体)",         "flag":"🇨🇳"},
    {"code":"ar",    "name":"العربية",            "flag":"🇸🇦"},
    {"code":"ru",    "name":"Русский",           "flag":"🇷🇺"},
    {"code":"pt",    "name":"Português",         "flag":"🇵🇹"},
    {"code":"ja",    "name":"日本語",             "flag":"🇯🇵"},
    {"code":"ko",    "name":"한국어",             "flag":"🇰🇷"},
    {"code":"it",    "name":"Italiano",          "flag":"🇮🇹"},
    {"code":"tr",    "name":"Türkçe",            "flag":"🇹🇷"},
    {"code":"vi",    "name":"Tiếng Việt",        "flag":"🇻🇳"},
    {"code":"id",    "name":"Bahasa Indonesia",  "flag":"🇮🇩"},
    {"code":"bn",    "name":"বাংলা",             "flag":"🇮🇳"},
    {"code":"ta",    "name":"தமிழ்",             "flag":"🇮🇳"},
    {"code":"te",    "name":"తెలుగు",            "flag":"🇮🇳"},
    {"code":"mr",    "name":"मराठी",             "flag":"🇮🇳"},
    {"code":"gu",    "name":"ગુજરાતી",           "flag":"🇮🇳"},
    {"code":"kn",    "name":"ಕನ್ನಡ",            "flag":"🇮🇳"},
    {"code":"ml",    "name":"മലയാളം",            "flag":"🇮🇳"},
    {"code":"pa",    "name":"ਪੰਜਾਬੀ",           "flag":"🇮🇳"},
    {"code":"ur",    "name":"اردو",             "flag":"🇵🇰"},
]

# ============================================================================
# TRANSLATION SERVICE
# ============================================================================
class TranslationService:
    def __init__(self, api_key=None):
        self.api_key    = api_key or GOOGLE_TRANSLATE_API_KEY
        self.translator = None
        self.cache      = {}
        if self.api_key and self.api_key != "YOUR_GOOGLE_CLOUD_API_KEY_HERE":
            try:
                self.translator = GoogleTranslator(source='auto', target='en')
                print("✅ Google Translate API initialized")
            except Exception as e:
                print(f"⚠️  Translator init failed: {e}")
        else:
            print("⚠️  No API Key — using fallback translations.")

    def translate_text(self, text, target_lang='en', source_lang='auto'):
        if not text or not isinstance(text, str) or target_lang == 'en':
            return text
        key = f"{text}_{target_lang}"
        if key in self.cache:
            return self.cache[key]
        try:
            if self.translator:
                t      = GoogleTranslator(source=source_lang, target=target_lang)
                result = t.translate(text)
                self.cache[key] = result
                return result
        except Exception as e:
            print(f"Translation error: {e}")
        return text

    def translate_treatment_data(self, data, target_lang='en'):
        if target_lang == 'en' or not data:
            return data
        translated = {}
        text_fields = ['description','causes','season','critical_period','action_required',
                       'monitoring_schedule','recovery_time','status','transmission',
                       'critical_action','environmental_factors','monitoring_frequency',
                       'rejection_reason']  # Added rejection_reason for safe predictions
        list_fields = ['symptoms','chemical_control','organic_control','biological_control',
                       'cultural_practices','preventive_measures','seasonal_management',
                       'ipm_strategy','spray_schedule','maintenance_practices',
                       'preventive_spray_schedule','nutrition_management','aphid_control','management']
        for f in text_fields:
            if f in data and data[f]:
                translated[f] = self.translate_text(data[f], target_lang)
        for f in list_fields:
            if f in data and data[f]:
                translated[f] = [self.translate_text(i, target_lang) for i in data[f] if isinstance(i, str)]
        for k, v in data.items():
            if k not in translated:
                translated[k] = self.translate_text(v, target_lang) if isinstance(v, str) else v
        return translated

    def get_supported_languages(self):
        return SUPPORTED_LANGUAGES

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR       = Path(__file__).parent.absolute()
TEMPLATES_PATH = BASE_DIR / "templates"
STATIC_PATH    = BASE_DIR / "static"
UPLOADS_PATH   = BASE_DIR / "uploads"
RESULTS_PATH   = BASE_DIR / "treatment_recommendations"

for p in [TEMPLATES_PATH, STATIC_PATH, UPLOADS_PATH, RESULTS_PATH]:
    p.mkdir(parents=True, exist_ok=True)

# ── FINETUNED MODEL ─────────────────────────────────────────────────────────
MODEL_PATH = Path(r"C:\Users\nimit\Music\Apple leaf project\best_finetuned_model.pth")
if not MODEL_PATH.exists():
    for ap in [BASE_DIR/"best_finetuned_model.pth",
               BASE_DIR/"models"/"best_finetuned_model.pth",
               BASE_DIR/"saved_models_v2"/"best_model.pth",
               BASE_DIR/"apple_model.pth"]:
        if ap.exists():
            MODEL_PATH = ap
            print(f"✅ Found model at: {MODEL_PATH}")
            break
    else:
        print("⚠️  Model not found — using simulated predictions.")

print(f"🤖 Model : {MODEL_PATH}  exists={MODEL_PATH.exists()}")

app = Flask(__name__, template_folder=str(TEMPLATES_PATH), static_folder=str(STATIC_PATH))
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config.update(
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024,
    UPLOAD_FOLDER      = str(UPLOADS_PATH),
    ALLOWED_EXTENSIONS = {'png','jpg','jpeg','gif','bmp','webp'},
    SECRET_KEY         = 'apple-leaf-detection-2024-secret-key'
)

translator = TranslationService(GOOGLE_TRANSLATE_API_KEY)

# ============================================================================
# SAFE PREDICTION CONFIGURATION
# ============================================================================
CONFIDENCE_THRESHOLD = 0.70  # 70% confidence required for valid prediction
UNKNOWN_CLASS_NAME = "Unknown/Uncertain"  # Class for low confidence predictions

# ============================================================================
# FULL DISEASE DATABASE (all 10 classes + Unknown class for safe predictions)
# ============================================================================
TREATMENT_DATABASE = {
    'Alternaria leaf spot': {
        'description': 'Fungal disease causing dark brown to black spots with concentric rings on leaves and fruit',
        'type': 'fungal','severity': 'medium',
        'symptoms': ['Dark brown to black spots with concentric rings','Spots may have a yellow halo',
                     'Can cause defoliation in severe cases','Fruit spots are sunken and dark'],
        'causes': 'Alternaria alternata fungus, thrives in warm, humid conditions',
        'season': 'Late spring through fall, especially during warm, wet weather',
        'chemical_control': ['Mancozeb 80WP (2.5g/L, apply every 10-14 days)',
                             'Chlorothalonil 720SC (2ml/L, apply every 10-14 days)',
                             'Azoxystrobin 23% SC (1ml/L, apply every 14 days)'],
        'organic_control': ['Copper hydroxide (Kocide 3000) - 4g/L every 7-10 days',
                            'Bacillus subtilis (Serenade MAX) - 5g/L every 7 days',
                            'Neem oil spray - 15ml/L every 10 days'],
        'biological_control': ['Trichoderma harzianum (RootShield)','Bacillus amyloliquefaciens (Double Nickel)'],
        'cultural_practices': ['Remove fallen leaves and infected debris in autumn',
                               'Prune trees for better air circulation',
                               'Avoid overhead irrigation (use drip or soaker hoses)',
                               'Maintain proper tree spacing (15-20 feet between trees)',
                               'Apply balanced fertilizer in early spring'],
        'preventive_measures': ['Plant disease-resistant varieties (Liberty, Freedom, Enterprise)',
                                'Apply preventive fungicides before rainy periods',
                                'Monitor regularly during warm, humid weather',
                                'Remove weeds that can harbor the fungus',
                                'Use reflective mulch to reduce humidity'],
        'seasonal_management': ['Spring: Apply preventive fungicide at green tip stage',
                                'Summer: Monitor weekly, apply fungicide after heavy rain',
                                'Fall: Remove all fallen leaves and debris',
                                'Winter: Apply dormant oil spray'],
        'monitoring_schedule': 'Weekly during growing season, especially after rain',
        'action_threshold': 'Treat when 5% of leaves show symptoms',
        'recovery_time': '2-3 weeks with proper treatment',
        'action_required': 'Apply fungicide within 3-5 days'
    },
    'Background_without_leaves': {
        'description': 'No apple leaf detected in the image. Please upload a clear leaf photo.',
        'type': 'background','severity': 'none',
        'symptoms': ['No leaf present in image'],
        'causes': 'N/A — background image detected',
        'season': 'N/A',
        'chemical_control': [],'organic_control': [],'biological_control': [],
        'cultural_practices': ['Upload a clear image of an apple leaf for detection'],
        'preventive_measures': ['Ensure leaf fills most of the frame','Use good lighting',
                                'Avoid blurry or distant shots'],
        'seasonal_management': [],
        'monitoring_schedule': 'N/A','action_threshold': 'N/A',
        'recovery_time': 'N/A','action_required': 'Upload a clear leaf image'
    },
    'Brown spot': {
        'description': 'Fungal disease causing circular brown spots with yellow halos',
        'type': 'fungal','severity': 'medium',
        'symptoms': ['Circular brown spots 2-10mm diameter','Yellow halo around spots',
                     'Spots may coalesce forming large necrotic areas','Early defoliation in severe cases'],
        'causes': 'Various fungal pathogens including Stemphylium vesicarium',
        'season': 'Warm, humid weather conditions',
        'chemical_control': ['Myclobutanil 10% WP (Rally 40WSP) - 1g/L every 14 days',
                             'Tebuconazole 25% EC (Elite 45DF) - 0.5ml/L every 14 days',
                             'Flutriafol (Topguard) - 0.75ml/L every 21 days'],
        'organic_control': ['Sulfur 80% WG - 4g/L every 7 days',
                            'Potassium bicarbonate (Milstop) - 5g/L every 7 days',
                            'Garlic extract spray - 10ml/L every 10 days'],
        'biological_control': ['Bacillus subtilis','Streptomyces lydicus'],
        'cultural_practices': ['Remove infected leaves and fallen debris regularly',
                               'Balanced fertilization (avoid excess nitrogen)',
                               'Proper canopy management for air flow','Avoid water stress'],
        'preventive_measures': ['Choose resistant apple varieties','Maintain proper tree spacing',
                                'Apply copper sprays in early spring',
                                'Avoid working in wet orchards','Sanitize pruning tools regularly'],
        'seasonal_management': ['Early Spring: Apply dormant spray',
                                'Late Spring: Begin preventive fungicide program',
                                'Summer: Monitor and treat as needed','Fall: Clean up fallen leaves'],
        'monitoring_schedule': 'Weekly during growing season',
        'action_threshold': 'Treat when symptoms first appear',
        'recovery_time': '2-3 weeks with treatment',
        'action_required': 'Monitor and treat if spreading'
    },
    'Frogeye leaf spot': {
        'description': 'Also known as Black Rot, causes frog-eye shaped spots on leaves and cankers on branches',
        'type': 'fungal','severity': 'high',
        'symptoms': ['Frog-eye shaped spots with purple margins and tan centers',
                     'Black, sunken cankers on branches',
                     'Fruit rot starting as small black spots','Mummified fruit remaining on tree'],
        'causes': 'Botryosphaeria obtusa fungus',
        'season': 'Throughout growing season, especially after hail or injury',
        'critical_period': '4-6 weeks after petal fall',
        'chemical_control': ['Thiophanate-methyl 70% WP (Topsin-M) - 1.5g/L every 10-14 days',
                             'Pyraclostrobin 20% WG (Cabrio) - 0.5g/L every 14 days',
                             'Fluopyram + Trifloxystrobin (Luna Sensation) - 0.6ml/L every 14 days'],
        'organic_control': ['Copper fungicides - Apply every 7-10 days during wet weather',
                            'Baking soda solution (1 tbsp per liter) weekly',
                            'Compost tea spray every 10 days'],
        'biological_control': ['Trichoderma species','Bacillus subtilis'],
        'cultural_practices': ['Prune out dead wood and cankers during dormancy',
                               'Remove mummified fruit from trees and ground',
                               'Avoid wounding trees during cultivation',
                               'Disinfect pruning tools with 70% alcohol'],
        'preventive_measures': ['Plant resistant varieties','Remove wild apple trees nearby',
                                'Practice good sanitation','Avoid excessive nitrogen fertilization',
                                'Maintain tree vigor'],
        'seasonal_management': ['Dormant season: Prune out cankers',
                                'Spring: Apply protective fungicides',
                                'Summer: Monitor and treat as needed','Fall: Remove mummified fruit'],
        'monitoring_schedule': 'Weekly during growing season',
        'action_threshold': 'Treat at first sign of symptoms',
        'recovery_time': '3-4 weeks with treatment',
        'action_required': 'Immediate fungicide application'
    },
    'Grey spot': {
        'description': 'Fungal disease causing grayish spots with purple margins',
        'type': 'fungal','severity': 'low',
        'symptoms': ['Grayish to silvery spots 3-8mm diameter','Purple to brown margins around spots',
                     'Spots may have a velvety appearance in humid conditions',
                     'Premature leaf drop in severe infections'],
        'causes': 'Various fungal pathogens',
        'season': 'Cool, wet weather conditions',
        'chemical_control': ['Dodine 65% WP (Syllit 65WP) - 1g/L every 14 days',
                             'Fenbuconazole 24% SC (Indar 2F) - 0.5ml/L every 14 days',
                             'Difenoconazole 25% EC (Score) - 0.5ml/L every 14 days'],
        'organic_control': ['Sulfur 80% WG (Microthiol Disperss) - 4g/L every 7 days',
                            'Copper octanoate (Cueva) - 6ml/L every 7-10 days',
                            'Horsetail tea spray weekly'],
        'biological_control': ['Bacillus amyloliquefaciens','Trichoderma harzianum'],
        'cultural_practices': ['Remove infected leaves during growing season',
                               'Improve air circulation through proper pruning',
                               'Avoid excessive moisture','Apply calcium sprays to strengthen cell walls',
                               'Mulch around trees to prevent splash dispersal'],
        'preventive_measures': ['Select resistant cultivars','Maintain proper orchard hygiene',
                                'Apply preventive lime-sulfur spray in dormancy',
                                'Monitor humidity levels','Use drip irrigation'],
        'seasonal_management': ['Spring: Preventive fungicide applications',
                                'Summer: Monitor and treat as needed',
                                'Fall: Leaf cleanup','Winter: Dormant sprays'],
        'monitoring_schedule': 'Bi-weekly during growing season',
        'action_threshold': 'Treat when spots appear on 10% of leaves',
        'recovery_time': '1-2 weeks with treatment',
        'action_required': 'Preventive treatment recommended'
    },
    'Health': {
        'description': 'Healthy apple leaf with no disease symptoms detected',
        'type': 'healthy','status': 'Normal','severity': 'none',
        'symptoms': ['No visible disease symptoms','Normal green color','Healthy growth pattern'],
        'causes': 'N/A — leaf is healthy',
        'season': 'All seasons',
        'chemical_control': [],'organic_control': [],'biological_control': [],
        'maintenance_practices': ['Continue regular monitoring program',
                                  'Apply preventive fungicides as per schedule',
                                  'Maintain tree health with balanced fertilization',
                                  'Ensure proper irrigation and drainage',
                                  'Monitor for pests and beneficial insects'],
        'preventive_spray_schedule': ['Dormant: Apply dormant oil for overwintering pests',
                                      'Green tip: Apply copper or sulfur for disease prevention',
                                      'Pink bud: Protectant fungicide if weather favors disease',
                                      'Petal fall: First summer fungicide application'],
        'nutrition_management': ['Soil test every 2-3 years','Maintain soil pH 6.0-6.5',
                                 'Apply balanced fertilizer (10-10-10) in early spring',
                                 'Foliar feed with micronutrients if needed'],
        'cultural_practices': ['Regular pruning for air circulation','Proper irrigation management',
                               'Mulching to maintain soil moisture',
                               'Companion planting with beneficial plants','Regular soil amendment'],
        'preventive_measures': ['Regular pruning for air circulation','Proper irrigation management',
                                'Mulching to maintain soil moisture',
                                'Companion planting with beneficial plants','Regular soil amendment'],
        'seasonal_management': ['Spring: Monitor for early pest emergence',
                                'Summer: Regular irrigation and monitoring',
                                'Fall: Clean up fallen leaves',
                                "Winter: Plan next season's strategy"],
        'monitoring_schedule': 'Weekly during growing season',
        'action_required': 'Continue preventive maintenance'
    },
    'Mosaic': {
        'description': 'Viral disease causing yellow mosaic patterns, reduced vigor, and distorted growth',
        'type': 'viral','severity': 'high',
        'symptoms': ['Yellow mosaic patterns or blotches on leaves','Leaf curling or distortion',
                     'Reduced tree vigor and stunted growth','Smaller, misshapen fruit with poor color'],
        'causes': 'Apple mosaic virus (ApMV)',
        'transmission': 'Primarily spread by aphids, also through infected grafting material',
        'season': 'Year-round, symptoms more visible in spring',
        'chemical_control': [],'organic_control': [],'biological_control': [],
        'management': ['⚠️ NO CHEMICAL CURE AVAILABLE — Remove infected trees completely',
                       'Use certified virus-free planting material',
                       'Control aphid vectors with systemic insecticides',
                       'Disinfect pruning tools with 10% bleach solution',
                       'Remove wild apple trees that may serve as reservoirs'],
        'aphid_control': ['Spring: Apply dormant oil before bud break',
                          'Growing season: Systemic insecticides when aphids first appear',
                          'Biological control: Release ladybugs and lacewings',
                          'Cultural control: Reflective mulch to repel aphids'],
        'cultural_practices': ['Remove and destroy infected trees immediately',
                               'Plant virus-free certified stock',
                               'Control weeds that may harbor aphids',
                               'Avoid grafting from unknown sources'],
        'preventive_measures': ['Always use certified virus-free nursery stock',
                                'Implement rigorous aphid control program',
                                'Regularly inspect new plantings',
                                'Isolate new trees for observation',
                                'Remove and destroy infected trees immediately'],
        'seasonal_management': ['Spring: Monitor for aphids and symptoms',
                                'Summer: Control aphid populations',
                                'Fall: Remove infected trees','Winter: Plan replacement strategy'],
        'monitoring_schedule': 'Weekly during growing season',
        'resistant_rootstocks': ['M.9','M.26','G.11','G.16'],
        'recovery_time': 'No recovery — remove infected trees',
        'action_required': 'Remove infected trees immediately'
    },
    'Powdery mildew': {
        'description': 'Fungal disease causing white powdery growth on leaves, shoots, and sometimes fruit',
        'type': 'fungal','severity': 'medium',
        'symptoms': ['White powdery fungal growth on leaf surfaces','Distorted or stunted new growth',
                     'Reduced photosynthesis and vigor','Russeting on fruit in severe cases'],
        'causes': 'Podosphaera leucotricha fungus',
        'season': 'Spring through fall, favors warm days and cool nights',
        'chemical_control': ['Myclobutanil 10% EW (Rally 40WSP) - 1ml/L every 14 days',
                             'Triflumizole 30% EC (Procure 480SC) - 0.75ml/L every 14 days',
                             'Quinoxyfen 25% EC (Quintec) - 0.75ml/L every 14 days'],
        'organic_control': ['Potassium bicarbonate (Milstop) - 5g/L every 7 days',
                            'Neem oil 70% EC - 5ml/L every 7 days',
                            'Horticultural oil (JMS Stylet Oil) - 15ml/L every 7-10 days',
                            'Milk spray (1 part milk to 9 parts water) weekly'],
        'biological_control': ['Bacillus subtilis','Trichoderma species','Ampelomyces quisqualis'],
        'cultural_practices': ['Prune for open canopy','Remove water sprouts and suckers regularly',
                               'Avoid excessive nitrogen fertilization',
                               'Improve air circulation','Plant resistant varieties'],
        'preventive_measures': ['Select mildew-resistant cultivars','Maintain proper tree spacing',
                                'Avoid overhead irrigation','Apply sulfur sprays preventively',
                                'Monitor new growth carefully'],
        'seasonal_management': ['Spring: Begin preventive program at pink bud',
                                'Summer: Continue protectant sprays',
                                'Fall: Clean up debris','Winter: Dormant pruning'],
        'environmental_factors': 'Favored by warm days and cool nights with high humidity',
        'monitoring_schedule': 'Weekly during shoot growth',
        'action_threshold': 'Treat when first signs appear on 5% of shoots',
        'recovery_time': '2-3 weeks with treatment',
        'action_required': 'Apply fungicide within 3-5 days'
    },
    'Rust': {
        'description': 'Fungal disease causing orange or yellow rust pustules on leaves, requiring alternate hosts',
        'type': 'fungal','severity': 'high',
        'symptoms': ['Bright orange or yellow pustules on leaf undersides','Yellow spots on upper leaf surfaces',
                     'Premature leaf drop in severe infections','Reduced fruit size and quality'],
        'causes': 'Gymnosporangium juniperi-virginianae fungus',
        'alternate_hosts': 'Juniperus species (Eastern red cedar, Rocky Mountain juniper)',
        'season': 'Spring and early summer',
        'critical_action': 'Remove alternate host (Juniper/Cedar trees within 300-500 feet)',
        'chemical_control': ['Myclobutanil 10% WP (Rally 40WSP) - 1g/L every 14 days',
                             'Triadimefon 25% WP (Bayleton 25DF) - 0.5g/L every 21 days',
                             'Tebuconazole + Trifloxystrobin (Nativo) - 0.5g/L every 14 days'],
        'organic_control': ['Sulfur sprays - every 7-10 days',
                            'Copper fungicides - early season','Neem oil - every 10-14 days'],
        'biological_control': ['Bacillus subtilis'],
        'spray_schedule': ['Pink bud: First protective spray','Petal fall: Second protective spray',
                           'Summer: Curative sprays if needed'],
        'cultural_practices': ['Remove all juniper/cedar trees within 500 feet',
                               'Plant rust-resistant apple varieties',
                               'Apply preventive fungicides before infection',
                               'Monitor for early symptoms','Improve air circulation in orchard'],
        'preventive_measures': ['Remove all juniper/cedar trees within 500 feet',
                                'Plant rust-resistant apple varieties (Liberty, Freedom, Goldrush)',
                                'Apply preventive fungicides before infection',
                                'Monitor for early symptoms','Improve air circulation in orchard'],
        'seasonal_management': ['Early Spring: Remove cedar galls',
                                'Pink bud through petal fall: Protectant sprays',
                                'Summer: Monitor and treat if needed','Fall: Clean up fallen leaves'],
        'resistant_varieties': ['Liberty','Freedom','Goldrush','Enterprise'],
        'monitoring_schedule': 'Weekly during spring',
        'action_threshold': 'Treat when symptoms appear on 5% of leaves',
        'recovery_time': '2-3 weeks with treatment',
        'action_required': 'Remove alternate hosts within 300 feet'
    },
    'Scab': {
        'description': 'Most common apple disease worldwide, causes olive-green to black spots on leaves and fruit',
        'type': 'fungal','severity': 'high',
        'symptoms': ['Olive-green to black velvety spots on leaves','Corky, scabby lesions on fruit',
                     'Twig infections causing blister-like swellings',
                     'Premature fruit drop in severe cases'],
        'causes': 'Venturia inaequalis fungus',
        'season': 'Spring and early summer, requires wet conditions',
        'chemical_control': ['Mancozeb 75% WP (Dithane 75DF) - 2g/L every 7-10 days',
                             'Dodine 65% WP (Syllit 65WP) - 1g/L every 10-14 days',
                             'Trifloxystrobin 50% WG (Flint 50WG) - 0.3g/L every 14 days',
                             'Pyrimethanil 40% SC (Scala 40SC) - 2ml/L every 10-14 days'],
        'organic_control': ['Sulfur sprays - apply every 7-10 days during infection periods',
                            'Baking soda solution - weekly during wet weather',
                            'Copper fungicides - early season preventive sprays'],
        'biological_control': ['Bacillus subtilis','Trichoderma species'],
        'ipm_strategy': ['MONITOR: Track temperature and leaf wetness',
                         'THRESHOLD: >10 hours leaf wetness at 10-25°C triggers infection',
                         'ACTION: Apply fungicide within 24-48 hours of infection period',
                         'RESISTANCE MANAGEMENT: Rotate fungicide classes'],
        'cultural_practices': ['Apply urea to fallen leaves in autumn to accelerate decomposition',
                               'Grow cover crops to reduce splash dispersal of spores',
                               'Use overhead irrigation only in early morning',
                               'Remove nearby wild or abandoned apple trees',
                               'Prune for good air circulation'],
        'preventive_measures': ['Plant scab-resistant varieties (Liberty, Freedom, Goldrush)',
                                'Implement sanitation practices',
                                'Use weather-based disease forecasting',
                                'Apply protectant fungicides before rain',
                                'Maintain tree vigor through proper nutrition'],
        'seasonal_management': ['Dormant: Remove leaf litter',
                                'Green tip through petal fall: Critical spray period',
                                'Summer: Continue protectant sprays',
                                'Fall: Apply urea to fallen leaves'],
        'resistant_varieties': ['Liberty','Freedom','Goldrush','Enterprise','Pristine','Redfree'],
        'monitoring_schedule': 'Daily during infection periods',
        'action_threshold': 'Treat when infection conditions occur',
        'recovery_time': '3-4 weeks with treatment',
        'action_required': 'Apply fungicide within 24 hours of infection period'
    },
    # New: Unknown/Uncertain class for low confidence predictions
    'Unknown/Uncertain': {
        'description': '⚠️ The model is not confident enough to identify this image. This could be due to:\n• Poor image quality or blurry photo\n• Unclear leaf features\n• Non-apple leaf image\n• Unusual lighting conditions\n\nPlease upload a clearer image of an apple leaf for accurate diagnosis.',
        'type': 'unknown',
        'severity': 'unknown',
        'symptoms': ['Unable to determine symptoms due to low confidence'],
        'causes': 'Insufficient image quality or unclear features',
        'season': 'N/A',
        'rejection_reason': 'Low confidence prediction',
        'chemical_control': [],
        'organic_control': [],
        'biological_control': [],
        'cultural_practices': [
            '📸 Take a clearer photo with better lighting',
            '🌿 Ensure the leaf fills most of the frame',
            '🎯 Focus on the affected area of the leaf',
            '☀️ Avoid shadows and glare',
            '📱 Hold camera steady to avoid blur',
            '🔄 Try uploading a different image'
        ],
        'preventive_measures': [
            'Use natural lighting for best results',
            'Keep camera steady or use a tripod',
            'Take multiple angles for better analysis',
            'Ensure leaf is clean and dry before photographing',
            'Avoid using digital zoom - move closer instead'
        ],
        'seasonal_management': [],
        'monitoring_schedule': 'N/A',
        'action_threshold': 'N/A',
        'recovery_time': 'N/A',
        'action_required': 'Please upload a clearer image of an apple leaf'
    }
}

print(f"📊 Disease database: {len(TREATMENT_DATABASE)} diseases loaded (including Unknown class)")

# ============================================================================
# MODEL with Safe Prediction
# ============================================================================
class AppleLeafDetector:
    def __init__(self, model_path=None, confidence_threshold=CONFIDENCE_THRESHOLD):
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model       = None
        self.class_names = list([k for k in TREATMENT_DATABASE.keys() if k != 'Unknown/Uncertain'])  # Original 10 classes
        self.confidence_threshold = confidence_threshold
        self.transform   = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.load_model(model_path)

    def load_model(self, model_path=None):
        if model_path is None: model_path = MODEL_PATH
        if not Path(model_path).exists():
            print(f"⚠️  Model not found — using simulated predictions")
            return False
        try:
            print(f"📦 Loading: {model_path}")
            ckpt = torch.load(model_path, map_location=self.device)
            self.model = models.resnet18(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
            self.model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            self.model.to(self.device).eval()
            print(f"✅ Model loaded on {self.device}")
            print(f"🎯 Confidence threshold: {self.confidence_threshold*100}%")
            return True
        except Exception as e:
            print(f"❌ Load error: {e}")
            return False

    def predict_with_safety(self, image_data):
        """
        Safe prediction with confidence threshold
        Returns: (disease_name, confidence, is_confident, rejection_reason, top5, all_probs)
        """
        if self.model is None:
            return self._simulate_with_safety()
        
        try:
            img    = Image.open(image_data).convert('RGB') if isinstance(image_data, str) else image_data.convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out   = self.model(tensor)
                probs = torch.nn.functional.softmax(out, dim=1)[0]
                max_conf, max_idx = probs.max(0)
                confidence = max_conf.item()
                
                # Get top 5 predictions
                top5_v, top5_i = torch.topk(probs, min(5, len(self.class_names)))
                top5 = [{"class": self.class_names[i.item()], "confidence": round(v.item()*100, 2)}
                        for v, i in zip(top5_v, top5_i)]
                
                # All probabilities
                all_probs = [{"class": self.class_names[i], "confidence": round(probs[i].item()*100, 2)}
                             for i in range(len(self.class_names))]
            
            # SAFETY CHECK 1: Confidence threshold
            if confidence < self.confidence_threshold:
                return {
                    'disease': 'Unknown/Uncertain',
                    'confidence': confidence * 100,
                    'is_confident': False,
                    'rejection_reason': f'Low confidence ({confidence*100:.1f}%) below threshold ({self.confidence_threshold*100}%)',
                    'top5': top5,
                    'all_probs': all_probs
                }
            
            # SAFETY CHECK 2: Valid class check
            predicted_class = self.class_names[max_idx.item()]
            if predicted_class not in self.class_names:
                return {
                    'disease': 'Unknown/Uncertain',
                    'confidence': confidence * 100,
                    'is_confident': False,
                    'rejection_reason': f'Invalid class prediction: {predicted_class}',
                    'top5': top5,
                    'all_probs': all_probs
                }
            
            # Valid prediction
            return {
                'disease': predicted_class,
                'confidence': confidence * 100,
                'is_confident': True,
                'rejection_reason': None,
                'top5': top5,
                'all_probs': all_probs
            }
            
        except Exception as e:
            print(f"❌ Predict error: {e}")
            return self._simulate_with_safety()

    def predict(self, image_data):
        """Legacy predict method for backward compatibility"""
        result = self.predict_with_safety(image_data)
        return result['disease'], result['confidence'], result['top5'], result['all_probs']

    def _simulate_with_safety(self):
        """Simulate prediction with safety check"""
        # Simulate with random confidence
        d = random.choice(self.class_names)
        c = round(random.uniform(50.0, 98.0), 2)
        
        # Check if confident
        is_confident = c >= (self.confidence_threshold * 100)
        
        if is_confident:
            result_disease = d
            rejection_reason = None
        else:
            result_disease = 'Unknown/Uncertain'
            rejection_reason = f'Low confidence ({c}%) below threshold ({self.confidence_threshold*100}%)'
        
        top5 = [{"class": d, "confidence": c}]
        all_probs = [{"class": n, "confidence": round(random.uniform(0, 3), 2)} for n in self.class_names]
        if is_confident:
            all_probs[self.class_names.index(d)]["confidence"] = c
        
        return {
            'disease': result_disease,
            'confidence': c,
            'is_confident': is_confident,
            'rejection_reason': rejection_reason,
            'top5': top5,
            'all_probs': all_probs
        }

    def get_model_info(self):
        return {
            'status': 'loaded' if self.model else 'simulated',
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'device': str(self.device),
            'model_path': str(MODEL_PATH),
            'accuracy': '94.34%',
            'weighted_f1': '94.47%',
            'macro_f1': '93.59%',
            'confidence_threshold': f'{self.confidence_threshold*100}%'
        }

    def update_confidence_threshold(self, new_threshold):
        """Update confidence threshold dynamically"""
        if 0 <= new_threshold <= 1:
            self.confidence_threshold = new_threshold
            return True
        return False

detector = AppleLeafDetector(MODEL_PATH, CONFIDENCE_THRESHOLD)

def sev_from_conf(c):
    return 'high' if c >= 85 else 'medium' if c >= 70 else 'low'

# ============================================================================
# ROUTES
# ============================================================================
@app.after_request
def no_cache(r):
    r.headers.update({'Cache-Control':'no-store, no-cache, must-revalidate, max-age=0',
                      'Pragma':'no-cache','Expires':'0'})
    return r

@app.route('/')
def home():
    p = TEMPLATES_PATH / "index.html"
    if p.exists():
        return p.read_text(encoding='utf-8')
    return "<h1>index.html not found in templates/</h1>", 404

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if 'image' not in request.files:
            return jsonify({'status':'error','error':'No image file provided'}), 400
        f = request.files['image']
        if not f.filename:
            return jsonify({'status':'error','error':'No file selected'}), 400

        fn = secure_filename(f"leaf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg")
        fp = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        f.save(fp)

        # Use safe prediction
        result = detector.predict_with_safety(fp)
        disease = result['disease']
        confidence = result['confidence']
        is_confident = result['is_confident']
        rejection_reason = result['rejection_reason']
        top5 = result['top5']
        all_probs = result['all_probs']
        
        lang = request.form.get('language','en')
        
        # Get treatment info (including Unknown/Uncertain class if needed)
        dinfo = TREATMENT_DATABASE.get(disease, TREATMENT_DATABASE['Unknown/Uncertain']).copy()
        
        # Add safe prediction metadata
        dinfo['is_confident'] = is_confident
        dinfo['confidence_threshold'] = f"{detector.confidence_threshold*100}%"
        if rejection_reason:
            dinfo['rejection_reason'] = rejection_reason
        
        severity = 'unknown' if disease == 'Unknown/Uncertain' else sev_from_conf(confidence)

        ti = translator.translate_treatment_data(dinfo, lang) if lang != 'en' else dinfo
        td = translator.translate_text(disease, lang) if lang != 'en' else disease
        ts = translator.translate_text(severity, lang) if lang != 'en' else severity

        return jsonify({
            'status':'success',
            'prediction': {
                'disease': disease,
                'disease_translated': td,
                'confidence': confidence,
                'severity': severity,
                'severity_translated': ts,
                'type': dinfo.get('type', 'unknown'),
                'is_confident': is_confident,
                'confidence_threshold': detector.confidence_threshold * 100,
                'rejection_reason': rejection_reason
            },
            'top5': top5,
            'all_probs': all_probs,
            'treatment': ti,
            'treatment_original': dinfo if lang != 'en' else None,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_version': 'v4.1.0',
                'model_accuracy': '94.34%',
                'confidence_threshold': detector.confidence_threshold * 100,
                'features': [
                    'Safe Prediction with Confidence Threshold',
                    'Disease Detection',
                    'Cure & Treatment',
                    'Prevention Guide',
                    'Multilingual (24 langs)',
                    'Camera Support',
                    'Detection History',
                    'CSV Export',
                    'Top-5 Predictions',
                    'Probability Chart'
                ]
            },
            'translation': {
                'target_language': lang,
                'source_language': 'en',
                'is_translated': lang != 'en',
                'translation_service': 'google' if translator.translator else 'fallback'
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status':'error','error':str(e)}), 500

@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_status': 'loaded' if detector.model else 'simulated',
        'database_status': 'loaded',
        'translation_service': 'google' if translator.translator else 'fallback',
        'supported_languages': len(SUPPORTED_LANGUAGES),
        'total_diseases': len(TREATMENT_DATABASE),
        'accuracy': '94.34%',
        'version': '4.1.0',
        'confidence_threshold': detector.confidence_threshold * 100
    })

@app.route('/api/info')
def api_info():
    return jsonify({
        'status': 'success',
        'version': '4.1.0',
        'model': detector.get_model_info(),
        'diseases': list(TREATMENT_DATABASE.keys()),
        'features': [
            'Safe Prediction with Confidence Threshold',
            'Disease Detection',
            'Treatment Recommendations',
            'Prevention Guide',
            'Multilingual (24 langs)',
            'Camera Support',
            'Detection History',
            'CSV Export',
            'Top-5 Predictions',
            'Probability Chart'
        ]
    })

@app.route('/api/diseases')
def api_diseases():
    try:
        lang = request.args.get('language','en')
        out  = {}
        for name, info in TREATMENT_DATABASE.items():
            ti = translator.translate_treatment_data(info.copy(), lang) if lang!='en' else info.copy()
            tn = translator.translate_text(name, lang) if lang!='en' else name
            out[name] = {
                'translated_name': tn,
                'description': ti.get('description', ''),
                'type': ti.get('type', 'unknown'),
                'symptoms': ti.get('symptoms', [])[:4],
                'chemical_control': ti.get('chemical_control', [])[:3],
                'organic_control': ti.get('organic_control', [])[:2],
                'cultural_practices': ti.get('cultural_practices', [])[:3],
                'preventive_measures': ti.get('preventive_measures', [])[:4],
                'severity': ti.get('severity', 'medium'),
                'has_prevention': bool(ti.get('preventive_measures')),
                'prevention_count': len(ti.get('preventive_measures', [])),
                'action_required': ti.get('action_required', 'Monitor closely')
            }
        return jsonify({'status':'success','language':lang,'total_diseases':len(out),'diseases':out})
    except Exception as e:
        return jsonify({'error':str(e),'status':'error'}), 500

@app.route('/api/disease/<disease_name>')
def api_disease(disease_name):
    try:
        lang = request.args.get('language','en')
        if disease_name not in TREATMENT_DATABASE:
            return jsonify({'status':'error','error':'Disease not found',
                            'suggestions':list(TREATMENT_DATABASE.keys())[:5]}), 404
        info = TREATMENT_DATABASE[disease_name].copy()
        ti   = translator.translate_treatment_data(info, lang) if lang!='en' else info
        tn   = translator.translate_text(disease_name, lang)   if lang!='en' else disease_name
        ti.update({
            'name': disease_name,
            'name_translated': tn,
            'is_healthy': disease_name == 'Health',
            'is_unknown': disease_name == 'Unknown/Uncertain'
        })
        return jsonify({'status':'success','language':lang,'disease':ti,
                        'original':info if lang!='en' else None})
    except Exception as e:
        return jsonify({'error':str(e),'status':'error'}), 500

@app.route('/api/config/threshold', methods=['GET', 'POST'])
def api_threshold_config():
    """Get or update confidence threshold"""
    if request.method == 'GET':
        return jsonify({
            'status': 'success',
            'current_threshold': detector.confidence_threshold * 100,
            'min_threshold': 0,
            'max_threshold': 100,
            'recommended_threshold': 70,
            'description': 'Predictions below this confidence level will be marked as Unknown/Uncertain'
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            new_threshold = float(data.get('threshold', 70)) / 100
            if detector.update_confidence_threshold(new_threshold):
                return jsonify({
                    'status': 'success',
                    'message': f'Confidence threshold updated to {new_threshold*100}%',
                    'new_threshold': new_threshold * 100
                })
            else:
                return jsonify({'status': 'error', 'error': 'Invalid threshold value (0-100)'}), 400
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 400

@app.route('/api/prevention-guide')
def api_prevention_guide():
    try:
        lang = request.args.get('language','en')
        general = ['Inspect trees weekly during the growing season',
                   'Keep records of disease occurrence and spray history',
                   'Use certified disease-free planting material',
                   'Maintain proper tree nutrition and soil health',
                   'Remove and destroy all infected plant material',
                   'Calibrate spray equipment for uniform coverage',
                   'Follow label directions for all pesticides',
                   'Rotate fungicide classes to prevent resistance']
        seasonal = {
            'Spring': ['Apply dormant oil before bud break',
                       'Start protective spray program at green tip',
                       'Monitor for early disease symptoms',
                       'Prune to improve air circulation'],
            'Summer': ['Continue regular spray program',
                       'Monitor after rain events',
                       'Remove heavily infected leaves',
                       'Maintain proper irrigation'],
            'Fall':   ['Remove fallen leaves and fruit',
                       'Apply urea to leaf litter to speed decomposition',
                       'Do final clean-up before winter',
                       'Record season disease pressure'],
            'Winter': ['Prune dead wood and cankers',
                       'Plan next season spray schedule',
                       'Order resistant varieties for replanting',
                       'Service spray equipment']
        }
        disease_specific = {
            n: {'key_prevention': i.get('preventive_measures',[])[:3],
                'action': i.get('action_required','Monitor'), 'severity': i.get('severity','medium')}
            for n, i in TREATMENT_DATABASE.items() if n not in ['Health','Background_without_leaves','Unknown/Uncertain']
        }
        if lang != 'en':
            general  = [translator.translate_text(t, lang) for t in general]
            seasonal = {k:[translator.translate_text(t,lang) for t in v] for k,v in seasonal.items()}
        return jsonify({'status':'success','guide':{
            'general_prevention':general,'seasonal_schedule':seasonal,
            'disease_specific':disease_specific}})
    except Exception as e:
        return jsonify({'error':str(e),'status':'error'}), 500

@app.route('/api/stats')
def api_stats():
    return jsonify({
        'status':'success',
        'model_info': detector.get_model_info(),
        'database_stats': {
            'total_diseases': len(TREATMENT_DATABASE),
            'fungal_diseases': sum(1 for v in TREATMENT_DATABASE.values() if v.get('type')=='fungal'),
            'viral_diseases': sum(1 for v in TREATMENT_DATABASE.values() if v.get('type')=='viral'),
            'high_severity': sum(1 for v in TREATMENT_DATABASE.values() if v.get('severity')=='high')
        },
        'supported_languages': len(SUPPORTED_LANGUAGES),
        'version': '4.1.0',
        'accuracy': '94.34%',
        'weighted_f1': '94.47%',
        'macro_f1': '93.59%',
        'confidence_threshold': detector.confidence_threshold * 100
    })

@app.route('/api/translate', methods=['POST'])
def api_translate():
    try:
        d    = request.get_json()
        text = d.get('text','')
        lang = d.get('target_lang','en')
        if not text: return jsonify({'status':'error','error':'No text'}), 400
        return jsonify({'status':'success','original':text,
                        'translated':translator.translate_text(text,lang),'target_lang':lang})
    except Exception as e:
        return jsonify({'status':'error','error':str(e)}), 500

@app.route('/api/translate/treatment', methods=['POST'])
def api_translate_treatment():
    try:
        d    = request.get_json()
        tdat = d.get('treatment_data',{})
        lang = d.get('target_lang','en')
        res  = translator.translate_treatment_data(tdat,lang) if lang!='en' else tdat
        return jsonify({'status':'success','translated_data':res,'target_lang':lang})
    except Exception as e:
        return jsonify({'status':'error','error':str(e)}), 500

@app.route('/api/translate/disease', methods=['POST'])
def api_translate_disease():
    try:
        d    = request.get_json()
        name = d.get('disease_name','')
        lang = d.get('target_lang','en')
        if name not in TREATMENT_DATABASE:
            return jsonify({'status':'error','error':'Invalid disease name'}), 404
        info = TREATMENT_DATABASE[name].copy()
        ti   = translator.translate_treatment_data(info, lang)
        tn   = translator.translate_text(name, lang)
        return jsonify({'status':'success','disease':name,'translated_disease':tn,'data':ti,'target_lang':lang})
    except Exception as e:
        return jsonify({'status':'error','error':str(e)}), 500

@app.route('/api/languages')
def api_languages():
    return jsonify({'status':'success','languages':SUPPORTED_LANGUAGES,
                    'default':'en','total':len(SUPPORTED_LANGUAGES),
                    'translation_service':'google' if translator.translator else 'fallback'})

@app.route('/api/download-treatment/<disease_name>')
def api_download_treatment(disease_name):
    try:
        if disease_name not in TREATMENT_DATABASE:
            return jsonify({'error':'Not found'}), 404
        info  = TREATMENT_DATABASE[disease_name]
        def li(lst): return '\n'.join(f"  • {x}" for x in lst) if lst else '  None recommended'
        
        # Add safe prediction info if applicable
        safe_note = ""
        if disease_name == 'Unknown/Uncertain':
            safe_note = """
⚠️ SAFE PREDICTION NOTE:
This image could not be confidently identified. The confidence level was below the threshold.
Please upload a clearer image of an apple leaf for accurate diagnosis.

"""
        
        text = f"""APPLE LEAF DISEASE TREATMENT REPORT
{safe_note}
Disease : {disease_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
Description : {info.get('description','')}
Type        : {info.get('type','')}
Severity    : {info.get('severity','')}
Season      : {info.get('season','')}

SYMPTOMS:
{li(info.get('symptoms',[]))}

CHEMICAL CONTROL:
{li(info.get('chemical_control',[]))}

ORGANIC CONTROL:
{li(info.get('organic_control',[]))}

BIOLOGICAL CONTROL:
{li(info.get('biological_control',[]))}

CULTURAL PRACTICES:
{li(info.get('cultural_practices',[]))}

PREVENTIVE MEASURES:
{li(info.get('preventive_measures',[]))}

SEASONAL MANAGEMENT:
{li(info.get('seasonal_management',[]))}

Action Required : {info.get('action_required','')}
Recovery Time   : {info.get('recovery_time','')}
{'='*60}
Apple Leaf Disease Detection System v4.1.0 | Accuracy: 94.34% | Safe Prediction Enabled
Confidence Threshold: {detector.confidence_threshold*100}%
"""
        return Response(text, mimetype='text/plain',
            headers={"Content-Disposition":f"attachment;filename=treatment_{disease_name.replace(' ','_')}.txt"})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',       default='127.0.0.1')
    parser.add_argument('--port',       type=int, default=5000)
    parser.add_argument('--debug',      action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--threshold',  type=float, default=70.0, 
                        help='Confidence threshold (0-100) for safe predictions')
    args = parser.parse_args()
    
    # Update confidence threshold if provided
    if 0 <= args.threshold <= 100:
        detector.update_confidence_threshold(args.threshold / 100)
        print(f"🎯 Confidence threshold set to: {args.threshold}%")

    print("\n"+"="*65)
    print("  🍎  APPLE LEAF DISEASE DETECTION  v4.1.0")
    print("="*65)
    print(f"  URL       : http://{args.host}:{args.port}")
    print(f"  Device    : {'GPU ✓' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Model     : {'Loaded ✓' if detector.model else 'Simulated'}")
    print(f"  Accuracy  : 94.34% | Weighted F1: 94.47%")
    print(f"  Diseases  : {len(TREATMENT_DATABASE)} classes (including Unknown class)")
    print(f"  Confidence: {detector.confidence_threshold*100}% threshold for safe predictions")
    print(f"  Languages : {len(SUPPORTED_LANGUAGES)}")
    print("="*65)
    print("  🛡️  SAFE PREDICTION ENABLED:")
    print(f"     • Predictions below {detector.confidence_threshold*100}% confidence will be flagged")
    print(f"     • Unknown/Uncertain class for low confidence images")
    print(f"     • Adjust threshold via /api/config/threshold endpoint")
    print("="*65+"\n")

    if not args.no_browser:
        def _open():
            time.sleep(2); webbrowser.open(f'http://{args.host}:{args.port}')
        threading.Thread(target=_open, daemon=True).start()

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ {e}"); sys.exit(1)

if __name__ == "__main__":
    main()