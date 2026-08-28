"""
Milestone 1: Flask backend
- Serves a QR code that points the mobile browser to the garment selection page
- Receives the user's garment selection from mobile and stores it for the desktop client
"""

import io
import qrcode
from flask import Flask, jsonify, request, send_file, render_template_string

app = Flask(__name__)

# In-memory store for the demo. Replace with a real session/DB layer later.
STATE = {"selected_garment": None}

# Placeholder catalogue - replace with real data pulled from garments/ once populated
GARMENTS = [
    {"id": "shirt1", "name": "Blue Casual Shirt"},
    {"id": "shirt2", "name": "White Formal Shirt"},
    {"id": "jacket1", "name": "Black Jacket"},
]

MOBILE_PAGE = """
<!doctype html>
<title>Select a Garment</title>
<style>
  body { font-family: sans-serif; max-width: 400px; margin: 40px auto; padding: 0 16px; }
  h2 { text-align: center; }
  ul { list-style: none; padding: 0; }
  li { margin: 10px 0; }
  a { display: block; padding: 14px; text-align: center; background: #2f5fa8;
      color: white; text-decoration: none; border-radius: 8px; font-size: 16px; }
  a:active { background: #244a85; }
  #confirm { text-align: center; color: #2f5fa8; font-weight: bold; }
</style>
<h2>Choose a garment to try on</h2>
<ul>
{% for g in garments %}
  <li><a href="/select/{{ g.id }}">{{ g.name }}</a></li>
{% endfor %}
</ul>
<p id="confirm"></p>
"""

SELECT_CONFIRM_PAGE = """
<!doctype html>
<title>Selected</title>
<body style="font-family: sans-serif; text-align: center; margin-top: 60px;">
  <h2>Selected: {{ name }}</h2>
  <p>Look at the desktop screen now.</p>
</body>
"""

DESKTOP_PAGE = """
<!doctype html>
<title>Virtual Try-On - Desktop</title>
<style>
  body { font-family: sans-serif; text-align: center; margin-top: 40px; }
  #qr { border: 1px solid #ddd; padding: 12px; display: inline-block; }
  #status { font-size: 20px; margin-top: 24px; }
  #garment-name { color: #2f5fa8; font-weight: bold; }
</style>
<h1>Virtual Try-On</h1>
<p>Scan this QR code with your phone to pick a garment</p>
<div id="qr"><img src="/qr" alt="QR code" width="220" height="220"></div>
<div id="status">Waiting for garment selection&hellip;</div>

<script>
  async function poll() {
    try {
      const res = await fetch('/api/selection');
      const data = await res.json();
      const statusEl = document.getElementById('status');
      if (data.selected_garment) {
        statusEl.innerHTML = 'Selected garment: <span id="garment-name">' + data.selected_garment + '</span>';
      }
    } catch (e) {
      // server not reachable yet, keep polling
    }
  }
  setInterval(poll, 1500);
  poll();
</script>
"""


@app.route("/")
def desktop_home():
    return render_template_string(DESKTOP_PAGE)


@app.route("/qr")
def get_qr():
    # In production, replace with your machine's LAN IP so a phone can reach it
    mobile_url = request.host_url + "mobile"
    img = qrcode.make(mobile_url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/mobile")
def mobile_page():
    return render_template_string(MOBILE_PAGE, garments=GARMENTS)


@app.route("/select/<garment_id>")
def select_garment(garment_id):
    STATE["selected_garment"] = garment_id
    match = next((g for g in GARMENTS if g["id"] == garment_id), None)
    name = match["name"] if match else garment_id
    return render_template_string(SELECT_CONFIRM_PAGE, name=name)


@app.route("/api/selection")
def get_selection():
    # Desktop client polls this to know what the user picked on mobile
    return jsonify(STATE)


if __name__ == "__main__":
    # host="0.0.0.0" so it's reachable from a phone on the same network
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
