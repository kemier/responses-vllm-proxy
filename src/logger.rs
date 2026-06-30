use chrono::Local;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
use uuid::Uuid;

/// A single log record enqueued for the background writer task.
struct LogRecord {
    level: LogLevel,
    module: String,
    message: String,
}

/// Log levels
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
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
    #[allow(dead_code)]
    max_total_size_bytes: u64,
    max_file_size_bytes: u64,
    file: Option<std::fs::File>,
    current_size: u64,
    to_stderr: bool,
    level: LogLevel,
}

/// Sender to the background writer task. Set once during `init()`; reads are
/// a lock-free atomic via `OnceLock::get()`. Producers call `try_send(&self)`,
/// so no `Sender` clone is needed on the hot path.
static SENDER: OnceLock<mpsc::Sender<LogRecord>> = OnceLock::new();

/// Current log level, captured at `init()` time. Read via a relaxed atomic
/// load so producers can cheaply skip filtered records before any allocation.
static CURRENT_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Info as u8);

static CLEANER_RUNNING: AtomicBool = AtomicBool::new(false);

/// Count of log records dropped because the channel was full.
/// Periodically reported instead of emitting one stderr warning per drop,
/// which under heavy load makes the problem worse than the drop itself.
static DROPPED_COUNT: AtomicU64 = AtomicU64::new(0);

/// Log directory captured at `init()` time, used by `error_request` to
/// place per-request error files alongside main.log. Read-only after init.
static LOG_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Whether to also emit log lines to stderr. Captured at `init()` time.
static TO_STDERR: OnceLock<bool> = OnceLock::new();

/// Initialize the logger.
///
/// Spawns a dedicated tokio task that owns the log `File`. Producers send
/// `(level, module, message)` records through a bounded `mpsc::channel`;
/// the public logging functions are non-blocking. If the channel is full,
/// the message is dropped and a one-line stderr warning is emitted to avoid
/// backpressure stalls that would serialize concurrent request handlers.
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
    CURRENT_LEVEL.store(level as u8, Ordering::Relaxed);

    let mut state = LoggerState {
        log_dir: log_dir.clone(),
        max_total_size_bytes: max_total_size_mb * 1024 * 1024,
        max_file_size_bytes: 10 * 1024 * 1024, // 10 MB per file
        file,
        current_size,
        to_stderr,
        level,
    };

    // Bounded channel: 8192 in-flight records. Heavy debug traffic with many
    // concurrent codex clients can easily exceed 1024 in-flight records,
    // causing log drops. 8192 gives the writer task more headroom while
    // still capping memory.
    let (tx, mut rx) = mpsc::channel::<LogRecord>(8192);

    // Spawn the dedicated writer task that owns the File. Only this task
    // ever touches the file handle, so no mutex is needed on the hot path.
    tokio::spawn(async move {
        while let Some(rec) = rx.recv().await {
            write_record(&mut state, rec);
        }
    });

    let _ = SENDER.set(tx);
    let _ = LOG_DIR.set(log_dir.clone());
    let _ = TO_STDERR.set(to_stderr);

    // Start background cleaner (OS thread, independent of tokio runtime)
    if !CLEANER_RUNNING.swap(true, Ordering::SeqCst) {
        let dir = log_dir.clone();
        let max_bytes = max_total_size_mb * 1024 * 1024;
        std::thread::spawn(move || {
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
        std::thread::sleep(Duration::from_secs(60));
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

/// Write a single record to file and/or stderr. Called only by the
/// background writer task, so no locking is required.
fn write_record(state: &mut LoggerState, rec: LogRecord) {
    if rec.level > state.level {
        return;
    }

    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
    let level_str = format!("{:<5}", rec.level.as_str());
    let line = format!(
        "[{}] [--------] [{}] [{}] {}\n",
        timestamp, level_str, rec.module, rec.message
    );

    // Write to file
    if let Some(ref mut file) = state.file {
        if file.write_all(line.as_bytes()).is_ok() {
            let _ = file.flush();
            state.current_size += line.len() as u64;
            rotate_if_needed(state);
        }
    }

    // Write to stderr
    if state.to_stderr {
        let _ = io::stderr().write_all(line.as_bytes());
    }
}

/// Enqueue a log record without blocking. If the channel is full or not
/// initialized, the record is dropped and a one-line stderr warning is
/// emitted to avoid backpressure stalls.
fn enqueue(level: LogLevel, module: &str, message: &str) {
    // Cheap atomic check: skip all allocation if the level is filtered.
    if level as u8 > CURRENT_LEVEL.load(Ordering::Relaxed) {
        return;
    }

    if let Some(sender) = SENDER.get() {
        let record = LogRecord {
            level,
            module: module.to_string(),
            message: message.to_string(),
        };
        if sender.try_send(record).is_err() {
            // Channel full or closed — drop and count. Emit stderr only at
            // power-of-two drop counts (1, 2, 4, ... 1024, ...) so a flood
            // of drops doesn't itself become a perf problem.
            let n = DROPPED_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                let _ = writeln!(
                    io::stderr(),
                    "[{}] logger: channel full, dropped {} log record(s) total (latest: {} from {})",
                    Local::now().format("%Y-%m-%d %H:%M:%S"),
                    n,
                    level.as_str(),
                    module
                );
            }
        }
    } else {
        // Logger not initialized — emit directly to stderr as a fallback.
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(
            io::stderr(),
            "[{}] [--------] [{}] [{}] {}",
            timestamp,
            format_args!("{:<5}", level.as_str()),
            module,
            message
        );
    }
}

/// Core log function (non-blocking).
pub fn log(level: LogLevel, module: &str, message: &str) {
    enqueue(level, module, message);
}

/// Cheap lock-free check whether logging at `level` is enabled. Use to
/// guard expensive log-message construction on hot paths — e.g.:
///
/// ```ignore
/// if logger::is_enabled(logger::LogLevel::Debug) {
///     let preview: String = chunk.chars().take(300).collect();
///     logger::debug("ws", &format!("[{}] line: {}", id, preview));
/// }
/// ```
///
/// The atomic load here is the same one `enqueue` performs internally,
/// but exposing it publicly lets callers skip building the format string
/// and any preview/signature allocations when the record would be dropped
/// by the level filter anyway.
#[inline]
pub fn is_enabled(level: LogLevel) -> bool {
    (level as u8) <= CURRENT_LEVEL.load(Ordering::Relaxed)
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

/// Write an error log for a specific request (like Go's error-*.log files).
///
/// The error-specific file is written synchronously here (it is on the
/// error path, not the hot path), while the summary line is enqueued to
/// the background writer for main.log.
pub fn error_request(endpoint: &str, request_body: &str, response_body: &str, status: u16) {
    let ts = Local::now().format("%Y-%m-%dT%H%M%S");
    let id = &Uuid::new_v4().simple().to_string()[..8];
    let filename = format!("error-{}-{}.log", endpoint.replace('/', "-"), ts);

    let filepath = LOG_DIR.get().map(|d| d.join(&filename)).unwrap_or_else(|| PathBuf::from("logs").join(&filename));

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
    if *TO_STDERR.get().unwrap_or(&true) {
        let _ = io::stderr().write_all(format!(
            "[{}] [error] [request] {} returned {}: see {}\n",
            Local::now().format("%Y-%m-%d %H:%M:%S"),
            endpoint, status, filename
        ).as_bytes());
    }

    // Write error file (synchronous — error path only)
    if let Ok(mut f) = OpenOptions::new().create(true).write(true).truncate(true).open(&filepath) {
        let _ = f.write_all(content.as_bytes());
    }

    // Enqueue summary line to main.log (non-blocking)
    log(
        LogLevel::Error,
        "request",
        &format!("{} returned HTTP {} — saved to {}", endpoint, status, filename),
    );
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
