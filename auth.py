import streamlit as st
import datetime

COMPANY_DOMAIN = "media.net"


def ensure_user():

    # Load previously remembered email
    email_from_browser = st.query_params.get("user", None)
    if email_from_browser and "user_email" not in st.session_state:
        st.session_state.user_email = email_from_browser

    # If no stored email → show popup
    if "user_email" not in st.session_state:

        st.markdown("""
            <style>
                .popup-wrapper {
                    position: fixed;
                    top: 0; left: 0;
                    width: 100vw; height: 100vh;
                    background: rgba(0,0,0,0.6);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 999999;
                }
                .popup-box {
                    width: 350px;
                    background: #ffffff;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
                    text-align: center;
                }
                .popup-title {
                    font-size: 20px;
                    font-weight: 600;
                    margin-bottom: 10px;
                }
                .popup-text {
                    font-size: 14px;
                    margin-bottom: 20px;
                }
            </style>

            <div class="popup-wrapper">
                <div class="popup-box">
                    <div class="popup-title">Enter your company email</div>
                    <div class="popup-text">This helps us understand app usage.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # The input box appears *inside the popup* overlay
        email = st.text_input(" ", placeholder=f"name@{COMPANY_DOMAIN}")

        if email:
            if email.endswith(f"@{COMPANY_DOMAIN}"):

                st.session_state.user_email = email

                # Remember for next visits
                st.query_params(user=email)

                st.rerun()
            else:
                st.error(f"Please enter a valid @{COMPANY_DOMAIN} email")

        st.stop()

    # Optional logging
    log_usage(st.session_state.user_email)

    return st.session_state.user_email


def log_usage(email):
    """Log usage to CSV"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("usage_log.csv", "a") as f:
        f.write(f"{email},{timestamp}\n")
