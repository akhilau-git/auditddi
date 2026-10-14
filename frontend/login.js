const apiBase = new URL(document.body.dataset.apiBaseUrl || '/api', window.location.origin).toString().replace(/\/$/, '');
const form = document.getElementById('login-form');
const error = document.getElementById('login-error');
const bootstrapForm = document.getElementById('bootstrap-form');

async function existingSession() {
  const response = await fetch(`${apiBase}/auth/session`);
  if (!response.ok) return;
  const session = await response.json();
  if (session.user?.workspace_url) window.location.replace(session.user.workspace_url);
  if (!session.authentication_enabled) {
    document.getElementById('login-status').textContent = 'Sign-in is not enabled on this server. Configure AUDITDDI_AUTH_ENABLED=true and restart the deployment.';
    return;
  }
  if (session.setup_required) {
    document.getElementById('login-status').textContent = 'This is a new deployment. Create the first administrator account.';
    form.hidden = true;
    bootstrapForm.hidden = false;
  }
}

form?.addEventListener('submit', async (event) => {
  event.preventDefault(); error.textContent = '';
  const response = await fetch(`${apiBase}/auth/login`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({email: email.value.trim(), password: password.value}) });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) { error.textContent = payload.detail || 'Unable to sign in.'; return; }
  window.location.assign(payload.user.workspace_url);
});
bootstrapForm?.addEventListener('submit', async (event) => {
  event.preventDefault();
  const output = document.getElementById('bootstrap-error'); output.textContent = '';
  const response = await fetch(`${apiBase}/auth/bootstrap`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({email:document.getElementById('bootstrap-email').value.trim(), password:document.getElementById('bootstrap-password').value, bootstrap_token:document.getElementById('bootstrap-token').value})});
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) { output.textContent = payload.detail || 'Administrator account could not be created.'; return; }
  window.location.assign('/admin.html');
});
existingSession();
