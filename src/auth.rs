use crate::logger;
use chrono::DateTime;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::sync::Mutex;

const TOKEN_ENDPOINT: &str =
    "https://login.microsoftonline.com/92e84ceb-fbfd-47ab-be52-080c6b87953f/oauth2/v2.0/token";
const CLIENT_ID: &str = "b46aa582-485a-4a7d-b30e-552dbd790b16";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenData {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: String,
    pub last_refresh: Option<String>,
}

pub struct TokenManager {
    auth_dir: PathBuf,
    /// Held for the entire check-or-refresh path in `get_token()`, so that
    /// only one task at a time can call the Microsoft token endpoint. This
    /// deduplicates concurrent refreshes when many clients hit the proxy
    /// simultaneously after the cached token expires.
    token: Mutex<Option<TokenData>>,
    client: reqwest::Client,
}

impl TokenManager {
    pub fn new(auth_dir: &str) -> Self {
        let path = PathBuf::from(auth_dir).join("ericai-ericai-user.json");
        logger::info("auth", &format!("Token file: {:?}", path));
        let token = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());
        Self {
            auth_dir: PathBuf::from(auth_dir),
            token: Mutex::new(token),
            client: reqwest::Client::new(),
        }
    }

    /// Forces an immediate token refresh regardless of expiry (call on upstream 401).
    pub async fn force_refresh(&self) -> Result<String, String> {
        let mut guard = self.token.lock().await;
        self.refresh_locked(&mut guard).await?;
        guard
            .as_ref()
            .map(|t| t.access_token.clone())
            .ok_or_else(|| "No token available after force-refresh".to_string())
    }

    /// Returns a valid access token, refreshing it if necessary.
    ///
    /// The mutex is held across the whole check-or-refresh path so that
    /// concurrent callers serialize on a single refresh: the first task to
    /// discover an expired token performs the HTTP refresh while the rest
    /// simply pick up the freshly-cached result.
    pub async fn get_token(&self) -> Result<String, String> {
        let mut guard = self.token.lock().await;

        // Fast path: cached token still valid.
        if let Some(t) = guard.as_ref() {
            if !is_expired(&t.expires_at) {
                return Ok(t.access_token.clone());
            }
        }

        // Slow path: refresh under the lock so concurrent callers dedupe.
        self.refresh_locked(&mut guard).await?;

        guard
            .as_ref()
            .map(|t| t.access_token.clone())
            .ok_or_else(|| "No token available".to_string())
    }

    /// Performs the token refresh. The caller already holds the lock.
    async fn refresh_locked(&self, guard: &mut Option<TokenData>) -> Result<(), String> {
        let refresh_token = guard
            .as_ref()
            .and_then(|t| t.refresh_token.clone())
            .ok_or_else(|| "No refresh token".to_string())?;

        logger::info("auth", "Refreshing EricAI access token...");
        let scope = format!("api://{}/API offline_access", CLIENT_ID);
        let params = [
            ("client_id", CLIENT_ID),
            ("grant_type", "refresh_token"),
            ("refresh_token", &refresh_token),
            ("scope", &scope),
        ];

        let resp = self
            .client
            .post(TOKEN_ENDPOINT)
            .form(&params)
            .send()
            .await
            .map_err(|e| format!("Refresh request failed: {}", e))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Refresh failed: {}", body));
        }

        #[derive(Deserialize)]
        struct TokenResponse {
            access_token: String,
            refresh_token: Option<String>,
            expires_in: i64,
        }

        let tr: TokenResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse token response: {}", e))?;

        let expires_at = chrono::Utc::now()
            .checked_add_signed(chrono::Duration::seconds(tr.expires_in))
            .unwrap_or_else(|| chrono::Utc::now() + chrono::Duration::hours(1))
            .to_rfc3339();

        let new_token = TokenData {
            access_token: tr.access_token,
            refresh_token: tr.refresh_token.or(Some(refresh_token)),
            expires_at: expires_at.clone(),
            last_refresh: Some(chrono::Utc::now().to_rfc3339()),
        };

        // Persist to disk
        let path = self.auth_dir.join("ericai-ericai-user.json");
        if let Ok(json) = serde_json::to_string_pretty(&new_token) {
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, &json).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }

        *guard = Some(new_token);
        logger::info(
            "auth",
            &format!("EricAI token refreshed (expires_at={})", expires_at),
        );
        Ok(())
    }
}

fn is_expired(expires_at: &str) -> bool {
    if let Ok(dt) = DateTime::parse_from_rfc3339(expires_at) {
        // Refresh 5 minutes before expiry
        dt < chrono::Utc::now() + chrono::Duration::minutes(5)
    } else {
        true
    }
}
