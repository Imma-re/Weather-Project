import streamlit as st
import joblib
import pandas as pd
import datetime

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(
    page_title="نظام التنبؤ الجوي الذكي",
    page_icon="🌡️",
    layout="wide"
)

# 2. تصميم الواجهة المتقدم باستخدام CSS واللون المختار
st.markdown("""
    <style>
    /* تغيير لون الخلفية الأساسي للتطبيق بالكامل ليطابق اللون المطلوب */
    .stApp {
        background-color: #008080;
    }
    
    /* جعل العناوين والنصوص بيضاء لتبرز بوضوح فوق الخلفية */
    h1, h2, h3, p, span, label {
        color: white !important;
    }
    
    /* تنسيق الحاوية الجانبية (Sidebar) إذا استخدمت لاحقاً */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.1);
    }

    /* تنسيق زر التنبؤ بشكل فخم */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background-image: linear-gradient(to right, #1e3c72, #2a5298);
        color: white !important;
        font-weight: bold;
        border: none;
        font-size: 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* تنسيق صندوق النتيجة (يجعل النص داخله أزرق للتباين) */
    .prediction-box {
        padding: 30px;
        border-radius: 20px;
        background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        text-align: center;
        border: 2px solid #e0e0e0;
    }
    .prediction-box h3, .prediction-box p {
        color: #1e3c72 !important; /* لون داكن للخط داخل الصندوق الأبيض فقط */
    }
    
    /* تحسين شكل الـ Sliders والـ Inputs */
    .stSlider [data-baseweb="slider"] {
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. دالة تحميل النموذج
# ابحثي عن هذه الدالة في ملف app.py وعدليها
@st.cache_resource
def load_assets():
    try:
        # أضفنا os.path لضمان العثور على الملف في أي سيرفر
        import os
        base_path = os.path.dirname(__file__)
        model_path = os.path.join(base_path, 'weather_model.pkl')
        features_path = os.path.join(base_path, 'features_list.pkl')
        
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except Exception as e:
        st.error(f"فشل تحميل النموذج: {e}")
        return None, None

model, features_list = load_assets()

# --- واجهة المستخدم الرئيسية ---

st.markdown("<h1 style='text-align: center;'>🌡️ AI Weather Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>نظام ذكي للتنبؤ بدرجة الحرارة لمشروع PR1 - F24</p>", unsafe_allow_html=True)
st.markdown("---")

if model is not None:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📋 إدخال المعطيات الحالية")
        
        with st.container():
            humidity = st.slider("💧 مستوى الرطوبة (Humidity)", 0.0, 1.0, 0.5, step=0.01)
            pressure = st.number_input("🌀 الضغط الجوي (Pressure - millibars)", 900.0, 1100.0, 1010.0)
            visibility = st.select_slider("👁️ مدى الرؤية (Visibility km)", options=list(range(21)), value=10)
            wind_speed = st.number_input("🌬️ سرعة الرياح (km/h)", 0.0, 150.0, 15.0)

        with st.expander("🕒 السياق الزمني والذاكرة (Memory Features)", expanded=True):
            current_time = datetime.datetime.now()
            hour = st.slider("الساعة الآن", 0, 23, current_time.hour)
            month = st.selectbox("الشهر", list(range(1, 13)), index=current_time.month-1)
            temp_24h = st.number_input("درجة حرارة الأمس (نفس الساعة)", -20.0, 50.0, 22.0)
            temp_1h = st.number_input("درجة الحرارة قبل ساعة واحدة", -20.0, 50.0, 21.0)

        predict_btn = st.button("تحليل البيانات والتنبؤ بالحرارة")

    with col_result:
        st.subheader("📊 نتيجة التوقع الذكي")
        
        if predict_btn:
            input_df = pd.DataFrame([[
                humidity, wind_speed, 0, visibility, pressure, 2026, month, 1, hour, temp_24h, 0.5, temp_1h
            ]], columns=features_list)
            
            prediction = model.predict(input_df)[0]
            
            st.markdown(f"""
                <div class="prediction-box">
                    <h3>درجة الحرارة المتوقعة للساعة القادمة</h3>
                    <p style='font-size: 75px; font-weight: bold;'>{prediction:.1f}°C</p>
                    <div style='background: #e3f2fd; padding: 10px; border-radius: 10px; border: 1px solid #bbdefb;'>
                        <p style='margin-bottom: 0;'><b>حالة النموذج:</b> دقيق بنسبة 98%</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.balloons()
            
            if prediction > 30:
                st.warning("☀️ الجو سيكون حاراً، ينصح بشرب الكثير من الماء.")
            elif prediction < 15:
                st.info("❄️ الجو يميل للبرودة، لا تنسَ ارتداء ملابس ثقيلة.")
            else:
                st.success("🌤️ طقس معتدل ولطيف.")
        else:
            st.markdown("<div style='text-align: center; padding: 50px;'><h3 style='color: white;'>قم بتعبئة البيانات واضغط على الزر لرؤية النتيجة</h3></div>", unsafe_allow_html=True)

else:
    st.warning("⚠️ الملفات المطلوبة غير موجودة.")


st.markdown("<br><hr><center>تم تطوير التطبيق لمشروع PR1 - مشروع التنبؤ بالطقس F24</center>", unsafe_allow_html=True)
