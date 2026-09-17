from backend.auth import AuthStore


def test_auth_store_hashes_passwords_and_revokes_sessions(tmp_path):
    store = AuthStore(tmp_path)
    user = store.create_user('Owner@example.org', 'a-long-local-test-password', 'research')

    assert store.has_users()
    assert store.authenticate('owner@example.org', 'wrong-password') is None
    authenticated = store.authenticate('owner@example.org', 'a-long-local-test-password')
    assert authenticated is not None
    assert authenticated['role'] == 'research'

    token = store.issue_session(user['id'])
    assert store.session_user(token)['email'] == 'owner@example.org'
    store.revoke_session(token)
    assert store.session_user(token) is None


def test_auth_store_deactivation_revokes_existing_sessions(tmp_path):
    store = AuthStore(tmp_path)
    user = store.create_user('reviewer@example.org', 'a-long-local-test-password', 'governance')
    token = store.issue_session(user['id'])

    store.set_user_active(user['id'], False)

    assert store.authenticate('reviewer@example.org', 'a-long-local-test-password') is None
    assert store.session_user(token) is None
    assert store.list_users()[0]['is_active'] == 0
