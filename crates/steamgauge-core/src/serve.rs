//! A local page that writes what it is given to a file, so a night's work is never only in a
//! browser.
//!
//! The adjudication page keeps answers in `localStorage`, which survives a closed tab and does
//! not survive a cleared cache, a private window, or a second machine. For a thousand questions
//! of somebody's own judgement that is the wrong place for the only copy, and asking them to
//! remember to press Export is asking them to be the backup.
//!
//! So the page is served from here instead of opened as a file. Every answer is posted as it is
//! made and lands on disk before the next question is drawn; the page reads the file back when
//! it opens, so the disk is the copy that matters and the browser is the cache. Nothing is
//! exported and nothing is lost.
//!
//! It is deliberately the smallest server that can do that: it binds to the loopback address,
//! answers three routes, and knows nothing about the page it is serving. There is no network
//! here to speak to and no second user to serve.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};

use crate::Result;

/// Bodies above this are refused rather than read. A thousand answers is well under a megabyte
/// and nothing else is expected to arrive.
const MOST_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug)]
pub struct Adjudication {
    page: String,
    answers: PathBuf,
}

impl Adjudication {
    #[must_use]
    pub fn new(page: String, answers: PathBuf) -> Self {
        Self { page, answers }
    }

    /// Serves until the process is stopped, printing the address to open.
    ///
    /// # Errors
    ///
    /// Fails if the port cannot be bound.
    pub fn serve(&self, port: u16, say: &dyn Fn(&str)) -> Result<()> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))?;
        let bound = listener.local_addr()?.port();
        say(&format!("adjudicating at http://127.0.0.1:{bound}/"));
        say(&format!(
            "answers are written to {} as they are made; stop with ctrl-c",
            self.answers.display()
        ));

        for stream in listener.incoming() {
            let Ok(mut stream) = stream else {
                continue;
            };
            // One bad request is one bad request, never the end of the sitting: the page is
            // the only client and a browser will retry.
            if let Err(problem) = self.answer(&mut stream) {
                say(&format!("a request failed: {problem}"));
            }
        }
        Ok(())
    }

    fn answer(&self, stream: &mut TcpStream) -> Result<()> {
        let Some((method, path, body)) = read_request(stream)? else {
            return Ok(());
        };
        match (method.as_str(), path.as_str()) {
            ("GET", "/") => reply(
                stream,
                200,
                "text/html; charset=utf-8",
                self.page.as_bytes(),
            ),
            ("GET", "/answers") => {
                let held = std::fs::read(&self.answers).unwrap_or_else(|_| b"[]".to_vec());
                reply(stream, 200, "application/json; charset=utf-8", &held)
            }
            ("POST", "/answers") => {
                self.keep(&body)?;
                reply(
                    stream,
                    200,
                    "application/json; charset=utf-8",
                    b"{\"saved\":true}",
                )
            }
            _ => reply(
                stream,
                404,
                "text/plain; charset=utf-8",
                b"no such thing here",
            ),
        }
    }

    /// Writes the answers beside the target and renames, so a crash mid-write cannot leave a
    /// half-written file where the whole adjudication used to be.
    fn keep(&self, body: &[u8]) -> Result<()> {
        serde_json::from_slice::<serde_json::Value>(body)?;
        if let Some(parent) = self
            .answers
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let beside = self.answers.with_extension("json.part");
        std::fs::write(&beside, body)?;
        std::fs::rename(&beside, &self.answers)?;
        Ok(())
    }
}

/// The method, the path and the body of one request, or `None` when the connection said nothing.
fn read_request(stream: &mut TcpStream) -> Result<Option<(String, String, Vec<u8>)>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut start = String::new();
    if reader.read_line(&mut start)? == 0 {
        return Ok(None);
    }
    let mut parts = start.split_whitespace();
    let method = parts.next().unwrap_or_default().to_owned();
    let path = parts.next().unwrap_or("/").to_owned();

    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some(said) = line.trim().strip_prefix("Content-Length:") {
            length = said.trim().parse().unwrap_or(0);
        }
    }
    if length > MOST_BYTES {
        return Ok(Some((method, path, Vec::new())));
    }

    let mut body = vec![0; length];
    if length > 0 {
        reader.read_exact(&mut body)?;
    }
    // The query string is not used by any route and would only ever be a way to reach one by
    // accident.
    let path = path.split('?').next().unwrap_or("/").to_owned();
    Ok(Some((method, path, body)))
}

fn reply(stream: &mut TcpStream, status: u16, kind: &str, body: &[u8]) -> Result<()> {
    let said = match status {
        200 => "OK",
        404 => "Not Found",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {said}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

/// Reads answers already on disk, for a caller that wants to say how many there are.
#[must_use]
pub fn answers_held(path: &Path) -> usize {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Vec<serde_json::Value>>(&bytes).ok())
        .map_or(0, |rows| rows.len())
}

#[cfg(test)]
mod tests {
    use super::Adjudication;

    #[test]
    fn a_posted_answer_lands_on_disk_whole_or_not_at_all() {
        let dir = crate::tempdir::Dir::new();
        let target = dir.path().join("gold-answers.json");
        let page = Adjudication::new("<p>hello</p>".to_owned(), target.clone());

        page.keep(br#"[{"subject":"verdict"}]"#)
            .expect("valid answers are written");
        assert_eq!(super::answers_held(&target), 1);

        // Nothing that is not JSON reaches the file, so a half-sent body cannot replace an
        // adjudication with a fragment of one.
        assert!(page.keep(b"{not json").is_err());
        assert_eq!(super::answers_held(&target), 1);
        assert!(!target.with_extension("json.part").exists());
    }

    #[test]
    fn answers_held_is_zero_when_there_is_no_file_yet() {
        let dir = crate::tempdir::Dir::new();
        assert_eq!(super::answers_held(&dir.path().join("nothing.json")), 0);
    }
}
