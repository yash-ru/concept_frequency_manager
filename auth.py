import streamlit as st
import datetime

COMPANY_DOMAIN = "media.net"


def ensure_user():
    """
    Handles:
    - Asking user for email once
    - Saving to session_state
    - Saving to query params (to remember)
    - Returning the valid user email
    """

    # Load from query params (if stored previously)
    email_from_browser = st.query_params.get("user", None)
    if email_from_browser and "user_email" not in st.session_state:
        st.session_state.user_email = email_from_browser

    # Ask for email if not stored
    if "user_email" not in st.session_state:

        # Popup UI
        st.markdown("""
            <style>
                .popup {
                    position: fixed;
                    top: 0; left: 0;
                    width: 100%; height: 100%;
                    background: rgba(0,0,0,0.6);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 99999;
                }
                .popup-box {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    width: 350px;
                    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
                    text-align: center;
                }
            </style>

            <div class="popup">
                <div class="popup-box">
                    <h3>Please enter your company email</h3>
                    <p>This helps us understand usage.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        email = st.text_input(" ", placeholder=f"name@{COMPANY_DOMAIN}")

        if email:
            if email.endswith(f"@{COMPANY_DOMAIN}"):
                st.session_state.user_email = email

                # Remember in browser
                st.query_params(user=email)

                st.rerun()
            else:
                st.error(f"Please enter a valid @{COMPANY_DOMAIN} email")

        st.stop()

    # Log the usage (optional)
    log_usage(st.session_state.user_email)

    return st.session_state.user_email


def log_usage(email):
    """Logs every access to a CSV file."""
    timestamp = datetime.datetime.now().isoformat()
    with open("usage_log.csv", "a") as f:
        f.write(f"{email},{timestamp}\n")
