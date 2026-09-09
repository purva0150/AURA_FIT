# Test access

Public demo: no authentication, accounts, passwords, or API keys are required.
Read the current preview origin from `frontend/.env` (`REACT_APP_BACKEND_URL`).
Create a disposable pairing session with `POST /api/sessions`; use its token in
`/mirror?s={token}` and `/remote?s={token}`. No credentials were created or modified.