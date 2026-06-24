use chrono::Local;
use once_cell::sync::Lazy;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

/// Log levels
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
}

impl LogLevel {
    fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
        }
    }

    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "error" => LogLevel::Error,
            "warn" | "warning" => LogLevel::Warn,
            "debug" => LogLevel::Debug,
            _ => LogLevel::Info,
        }
    }
}

struct LoggerState {
    log_dir: PathBuf,
    max_total_size_bytes: u64,
    max_file_size_bytes: u64,
    file: Option<std::fs::File>,
    current_size: u64,
    to_stderr: bool,
    level: LogLevel,
}

static STATE: Lazy<Mutex<LoggerState>> = Lazy::new(|| {
    Mutex::new(LoggerState {
        log_dir: PathBuf::from("logs"),
        max_total_size_bytes: 10 * 1024 * 1024, // 10 MB default
        max_file_size_bytes: 10 * 1024 * 1024,  // 10 MB per file
        file: None,
        current_size: 0,
        to_stderr: true,
        level: LogLevel::Info,
    })
});

static CLEANER_RUNNING: AtomicBool = AtomicBool::new(false);

/// Initialize the logger.
pub fn init(log_dir: &str, max_total_size_mb: u64, to_stderr: bool, level_str: &str) {
    let log_dir = PathBuf::from(log_dir);
    let _ = fs::create_dir_all(&log_dir);

    let main_log = log_dir.join("main.log");
    let current_size = fs::metadata(&main_log).map(|m| m.len()).unwrap_or(0);

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&main_log)
        .ok();

    let level = LogLevel::from_str(level_str);

    {
        let mut st = STATE.lock().unwrap();
        st.log_dir = log_dir.clone();
        st.max_total_size_bytes = max_total_size_mb * 1024 * 1024;
        st.file = file;
        st.current_size = current_size;
        st.to_stderr = to_stderr;
        st.level = level;
    }

    // Start background cleaner
    if !CLEANER_RUNNING.swap(true, Ordering::SeqCst) {
        let st = STATE.lock().unwrap();
        let dir = st.log_dir.clone();
        let max_bytes = st.max_total_size_bytes;
        drop(st);
        thread::spawn(move || {
            run_cleaner(dir, max_bytes);
        });
    }

    log(LogLevel::Info, "logger", &format!(
        "Logging initialized: dir={}, max_total={}MB, level={}",
        log_dir.display(),
        max_total_size_mb,
        level.as_str()
    ));
}

fn run_cleaner(log_dir: PathBuf, max_bytes: u64) {
    loop {
        thread::sleep(Duration::from_secs(60));
        if let Err(e) = enforce_size_limit(&log_dir, max_bytes) {
            eprintln!("logger: cleaner error: {}", e);
        }
    }
}

fn enforce_size_limit(log_dir: &Path, max_bytes: u64) -> io::Result<()> {
    if max_bytes == 0 {
        return Ok(());
    }

    let mut files: Vec<(PathBuf, u64, SystemTime)> = Vec::new();
    let mut total: u64 = 0;

    for entry in fs::read_dir(log_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_lowercase();
        if !name.ends_with(".log") && !name.ends_with(".log.gz") {
            continue;
        }
        let meta = entry.metadata()?;
        if !meta.is_file() {
            continue;
        }
        let path = entry.path();
        let size = meta.len();
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        files.push((path, size, mtime));
        total += size;
    }

    if total <= max_bytes {
        return Ok(());
    }

    // Sort oldest first
    files.sort_by_key(|(_, _, mtime)| *mtime);

    let protected = log_dir.join("main.log");

    let mut deleted = 0;
    for (path, size, _) in &files {
        if total <= max_bytes {
            break;
        }
        if *path == protected {
            continue; // Never delete main.log
        }
        if fs::remove_file(path).is_ok() {
            total -= *size;
            deleted += 1;
        }
    }

    if deleted > 0 {
        log(LogLevel::Debug, "logger", &format!(
            "Removed {} old log file(s) to enforce size limit", deleted
        ));
    }

    Ok(())
}

fn rotate_if_needed(state: &mut LoggerState) {
    if state.current_size < state.max_file_size_bytes {
        return;
    }

    // Close current file
    state.file = None;

    // Rotate: main.log → main-<timestamp>.log
    let main_log = state.log_dir.join("main.log");
    let ts = Local::now().format("%Y%m%dT%H%M%S");
    let rotated = state.log_dir.join(format!("main-{}.log", ts));
    let _ = fs::rename(&main_log, &rotated);

    // Open new file
    state.current_size = 0;
    state.file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&main_log)
        .ok();

    log(LogLevel::Debug, "logger", &format!(
        "Rotated main.log → {} (was {} bytes)",
        rotated.display(), state.max_file_size_bytes
    ));
}

/// Core log function.
pub fn log(level: LogLevel, module: &str, message: &str) {
    let mut st = STATE.lock().unwrap();
    if level > st.level {
        return;
    }

    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
    let level_str = format!("{:<5}", level.as_str());
    let line = format!("[{}] [--------] [{}] [{}] {}\n", timestamp, level_str, module, message);

    // Write to file
    if let Some(ref mut file) = st.file {
        if let Ok(()) = file.write_all(line.as_bytes()) {
            let _ = file.flush();
            st.current_size += line.len() as u64;
            rotate_if_needed(&mut st);
        }
    }

    // Write to stderr
    if st.to_stderr {
        let _ = io::stderr().write_all(line.as_bytes());
    }
}

/// Convenience macros/functions for each level
pub fn error(module: &str, msg: &str) {
    log(LogLevel::Error, module, msg);
}

pub fn warn(module: &str, msg: &str) {
    log(LogLevel::Warn, module, msg);
}

pub fn info(module: &str, msg: &str) {
    log(LogLevel::Info, module, msg);
}

#[allow(dead_code)]
pub fn debug(module: &str, msg: &str) {
    log(LogLevel::Debug, module, msg);
}

/// Write an error log for a specific request (like Go's error-*.log files)
pub fn error_request(endpoint: &str, request_body: &str, response_body: &str, status: u16) {
    let mut st = STATE.lock().unwrap();

    let ts = Local::now().format("%Y-%m-%dT%H%M%S");
    let id = &Uuid::new_v4().simple().to_string()[..8];
    let filename = format!("error-{}-{}.log", endpoint.replace('/', "-"), ts);
    let filepath = st.log_dir.join(&filename);

    let content = format!(
        "[{}] Error Log — {} (HTTP {})\n\
         Request ID: {}\n\
         Endpoint: {}\n\
         Status: {}\n\
         \n\
         --- Request Body ---\n{}\n\
         \n\
         --- Response Body ---\n{}\n",
        Local::now().format("%Y-%m-%d %H:%M:%S"),
        endpoint,
        status,
        id,
        endpoint,
        status,
        request_body,
        response_body,
    );

    // Write to stderr
    if st.to_stderr {
        let _ = io::stderr().write_all(format!(
            "[{}] [error] [request] {} returned {}: see {}\n",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            endpoint, status, filename
        ).as_bytes());
    }

    // Write error file
    if let Ok(mut f) = OpenOptions::new().create(true).write(true).truncate(true).open(&filepath) {
        let _ = f.write_all(content.as_bytes());
    }

    // Also write to main.log
    if let Some(ref mut file) = st.file {
        let line = format!(
            "[{}] [--------] [error ] [request] {} returned HTTP {} — saved to {}\n",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            endpoint, status, filename
        );
        let _ = file.write_all(line.as_bytes());
        let _ = file.flush();
        st.current_size += line.len() as u64;
        rotate_if_needed(&mut st);
    }
}

/// Extract a short error summary from an upstream response body.
/// Tries to parse JSON and extract .error.message; falls back to truncated text.
pub fn extract_error_summary(body: &str, max_len: usize) -> String {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        // Try common error paths
        if let Some(msg) = v.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str()) {
            return msg.to_string();
        }
        if let Some(msg) = v.get("error").and_then(|e| e.get("internal_message")).and_then(|m| m.as_str()) {
            return msg.to_string();
        }
        if let Some(msg) = v.get("message").and_then(|m| m.as_str()) {
            return msg.to_string();
        }
        if let Some(msg) = v.get("detail").and_then(|d| d.as_str()) {
            return msg.to_string();
        }
    }
    // Fallback: truncate on char boundary
    if body.len() <= max_len {
        body.to_string()
    } else {
        let mut end = max_len;
        while end > 0 && !body.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &body[..end])
    }
}
