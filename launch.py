import ssl

# Windows certificate stores occasionally contain malformed certs from OS updates.
# Python tries to load every cert in the store at ssl context creation time and
# crashes on the bad one. This skips malformed entries instead of aborting.
_orig_load_default_certs = ssl.SSLContext.load_default_certs

def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _orig_load_default_certs(self, purpose)
    except ssl.SSLError:
        pass

ssl.SSLContext.load_default_certs = _safe_load_default_certs

# Launch Streamlit in this same process (so the patch is already in place
# when tornado imports ssl on the next line).
from streamlit.web import cli as stcli
import sys

sys.argv = [
    "streamlit", "run", "app/main.py",
    "--server.maxUploadSize=2048",
    "--server.port=8501",
    "--server.enableXsrfProtection=false",
]
stcli.main()
