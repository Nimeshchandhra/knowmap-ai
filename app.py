
"""
Entry point: Login & Signup Page

FIX: Removed 'showNavigation' and added a CSS hack to hide the
sidebar. This is the correct, version-agnostic method.
"""

import streamlit as st
from auth_db import init_db, verify_user, add_user, create_token, get_user_profile

# ---------------- Streamlit config ----------------
# FIX: Removed invalid 'showNavigation=' argument
st.set_page_config(
    page_title="Login - KG Explorer", 
    layout="centered"
)

# FIX: Added CSS hack to hide the default page navigation
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

# ---------------- Auth UI ----------------
def login_signup_modal():
    st.markdown("""
    <style>
    .login-card {background-color:#1e1e1e;padding:2rem;border-radius:1rem;box-shadow:0 0 25px rgba(0,0,0,0.4);max-width:480px;margin:6vh auto;color:white}
    .stTextInput > div > div > input {background-color:#111 !important;color:white !important;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Semantic KG Explorer</h3>", unsafe_allow_html=True)
    st.markdown("Please sign in or create an account to continue")
    
    mode = st.radio("Mode", ["Login", "Signup"], horizontal=True, label_visibility="collapsed")
    
    with st.form(key=f"{mode}_form"):
        username = st.text_input("Username", placeholder="username")
        password = st.text_input("Password", type="password", placeholder="password")
        
        admin_flag = False
        if mode == "Signup":
            admin_flag = st.checkbox("Make admin account?")
        
        submitted = st.form_submit_button(mode)

        if submitted:
            if not username or not password:
                st.error("Please enter a username and password.")
                return

            if mode == "Login":
                user = verify_user(username, password)
                if user:
                    user_profile = get_user_profile(username)
                    
                    st.session_state["username"] = user_profile["username"]
                    st.session_state["display_name"] = user_profile.get("display_name") or user_profile["username"]
                    st.session_state["is_admin"] = user_profile["is_admin"]
                    st.session_state["theme"] = user_profile.get("theme", "light")
                    
                    st.success("Login successful! Redirecting...")
                    st.switch_page("pages/main_app.py")
                else:
                    st.error("Invalid credentials")
            
            elif mode == "Signup":
                try:
                    add_user(username, password, is_admin=admin_flag)
                    st.success("Account created. Please login.")
                except Exception as e:
                    st.error(f"Could not create account: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Entry point ----------------
if __name__ == "__main__":
    init_db()

    if "username" in st.session_state:
        st.switch_page("pages/main_app.py")
    else:
        login_signup_modal()
