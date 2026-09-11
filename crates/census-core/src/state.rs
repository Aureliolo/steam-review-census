//! Durable crawl state, so an interrupted corpus resumes instead of restarting.
//!
//! The shard is the unit of atomicity: it either finishes and its Parquet file is sealed,
//! or it is redone from the start. Resuming mid-file would mean trusting a Parquet footer
//! that was never written, and a shard is capped small enough that redoing one is cheap.

use std::{
    path::Path,
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, OptionalExtension, params};

use crate::{Result, shard::Shard};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS crawl (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id        INTEGER NOT NULL,
    snapshot_unix INTEGER NOT NULL,
    params_url    TEXT    NOT NULL,
    valve_total   INTEGER NOT NULL,
    finished_unix INTEGER
);
CREATE TABLE IF NOT EXISTS shard (
    crawl_id    INTEGER NOT NULL REFERENCES crawl(id),
    idx         INTEGER NOT NULL,
    start_date  INTEGER NOT NULL,
    end_date    INTEGER NOT NULL,
    expected    INTEGER NOT NULL,
    status      TEXT    NOT NULL,
    row_count   INTEGER NOT NULL DEFAULT 0,
    pages       INTEGER NOT NULL DEFAULT 0,
    stop_reason TEXT,
    max_created INTEGER,
    PRIMARY KEY (crawl_id, idx)
);
";

#[derive(Debug)]
pub struct CrawlState {
    conn: Mutex<Connection>,
}

#[derive(Debug, Clone, Copy)]
pub struct ShardRecord {
    pub idx: i64,
    pub shard: Shard,
}

/// A crawl that was interrupted and can be continued.
#[derive(Debug, Clone)]
pub struct ResumableCrawl {
    pub id: i64,
    pub snapshot_unix: i64,
    pub valve_total: u64,
    pub done_shards: usize,
    pub total_shards: usize,
}

impl CrawlState {
    /// # Errors
    ///
    /// Fails if the database cannot be created or the schema cannot be applied.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// # Errors
    ///
    /// Fails if the row cannot be written.
    pub fn begin_crawl(
        &self,
        app_id: u32,
        snapshot_unix: i64,
        params_url: &str,
        valve_total: u64,
    ) -> Result<i64> {
        let conn = self.lock();
        conn.execute(
            "INSERT INTO crawl (app_id, snapshot_unix, params_url, valve_total)
             VALUES (?1, ?2, ?3, ?4)",
            params![app_id, snapshot_unix, params_url, sql_i64(valve_total)],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// # Errors
    ///
    /// Fails if any shard row cannot be written.
    pub fn record_shards(&self, crawl_id: i64, shards: &[Shard]) -> Result<()> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        for (idx, shard) in shards.iter().enumerate() {
            tx.execute(
                "INSERT OR REPLACE INTO shard
                   (crawl_id, idx, start_date, end_date, expected, status)
                 VALUES (?1, ?2, ?3, ?4, ?5, 'pending')",
                params![
                    crawl_id,
                    i64::try_from(idx).unwrap_or(i64::MAX),
                    shard.start_date,
                    shard.end_date,
                    sql_i64(shard.expected)
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// The most recent unfinished crawl for an app, if there is one.
    ///
    /// # Errors
    ///
    /// Fails if the query cannot be run.
    pub fn resumable(&self, app_id: u32) -> Result<Option<ResumableCrawl>> {
        let conn = self.lock();
        let found = conn
            .query_row(
                "SELECT id, snapshot_unix, valve_total FROM crawl
                 WHERE app_id = ?1 AND finished_unix IS NULL
                 ORDER BY id DESC LIMIT 1",
                params![app_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .optional()?;

        let Some((id, snapshot_unix, valve_total)) = found else {
            return Ok(None);
        };
        let total_shards: i64 = conn.query_row(
            "SELECT COUNT(*) FROM shard WHERE crawl_id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        let done_shards: i64 = conn.query_row(
            "SELECT COUNT(*) FROM shard WHERE crawl_id = ?1 AND status = 'done'",
            params![id],
            |row| row.get(0),
        )?;
        Ok(Some(ResumableCrawl {
            id,
            snapshot_unix,
            valve_total: u64::try_from(valve_total).unwrap_or(0),
            done_shards: usize::try_from(done_shards).unwrap_or(0),
            total_shards: usize::try_from(total_shards).unwrap_or(0),
        }))
    }

    /// Shards still to do, with anything left mid-flight by a previous process reset.
    ///
    /// # Errors
    ///
    /// Fails if the query cannot be run.
    pub fn pending_shards(&self, crawl_id: i64) -> Result<Vec<ShardRecord>> {
        let conn = self.lock();
        // A shard marked running belongs to a process that is gone, and a half-written
        // Parquet file has no footer, so it starts over rather than resuming mid-file.
        conn.execute(
            "UPDATE shard SET status = 'pending' WHERE crawl_id = ?1 AND status = 'running'",
            params![crawl_id],
        )?;
        let mut stmt = conn.prepare(
            "SELECT idx, start_date, end_date, expected FROM shard
             WHERE crawl_id = ?1 AND status <> 'done'
             ORDER BY expected DESC",
        )?;
        let rows = stmt
            .query_map(params![crawl_id], |row| {
                Ok(ShardRecord {
                    idx: row.get(0)?,
                    shard: Shard {
                        start_date: row.get(1)?,
                        end_date: row.get(2)?,
                        expected: u64::try_from(row.get::<_, i64>(3)?).unwrap_or(0),
                    },
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// # Errors
    ///
    /// Fails if the row cannot be updated.
    pub fn mark_running(&self, crawl_id: i64, idx: i64) -> Result<()> {
        self.lock().execute(
            "UPDATE shard SET status = 'running' WHERE crawl_id = ?1 AND idx = ?2",
            params![crawl_id, idx],
        )?;
        Ok(())
    }

    /// # Errors
    ///
    /// Fails if the row cannot be updated.
    pub fn mark_done(
        &self,
        crawl_id: i64,
        idx: i64,
        rows: u64,
        pages: u32,
        stop_reason: &str,
        max_created: Option<i64>,
    ) -> Result<()> {
        self.lock().execute(
            "UPDATE shard SET status = 'done', row_count = ?3, pages = ?4,
                    stop_reason = ?5, max_created = ?6
             WHERE crawl_id = ?1 AND idx = ?2",
            params![
                crawl_id,
                idx,
                sql_i64(rows),
                i64::from(pages),
                stop_reason,
                max_created
            ],
        )?;
        Ok(())
    }

    /// # Errors
    ///
    /// Fails if the query cannot be run.
    pub fn completed_totals(&self, crawl_id: i64) -> Result<(u64, u32)> {
        let conn = self.lock();
        let (rows, pages): (i64, i64) = conn.query_row(
            "SELECT COALESCE(SUM(row_count), 0), COALESCE(SUM(pages), 0)
             FROM shard WHERE crawl_id = ?1 AND status = 'done'",
            params![crawl_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok((
            u64::try_from(rows).unwrap_or(0),
            u32::try_from(pages).unwrap_or(u32::MAX),
        ))
    }

    /// # Errors
    ///
    /// Fails if the query cannot be run.
    pub fn all_shards_done(&self, crawl_id: i64) -> Result<bool> {
        let conn = self.lock();
        let remaining: i64 = conn.query_row(
            "SELECT COUNT(*) FROM shard WHERE crawl_id = ?1 AND status <> 'done'",
            params![crawl_id],
            |row| row.get(0),
        )?;
        Ok(remaining == 0)
    }

    /// # Errors
    ///
    /// Fails if the rows cannot be updated.
    pub fn finish_crawl(&self, crawl_id: i64) -> Result<()> {
        self.lock().execute(
            "UPDATE crawl SET finished_unix = ?2 WHERE id = ?1",
            params![crawl_id, now_unix()],
        )?;
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        // A poisoned lock means another thread panicked mid-statement; the connection
        // itself is still usable and losing the crawl over it would be worse.
        self.conn
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// SQLite stores signed 64-bit integers only, so unsigned counts cross the boundary here.
/// Saturating is safe: a corpus large enough to overflow this does not exist.
fn sql_i64(n: u64) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

fn now_unix() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    )
    .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> (CrawlState, tempdir::Dir) {
        let dir = tempdir::Dir::new();
        let state = CrawlState::open(&dir.path().join("state.sqlite")).unwrap();
        (state, dir)
    }

    fn shards() -> Vec<Shard> {
        vec![
            Shard {
                start_date: 0,
                end_date: 99,
                expected: 10,
            },
            Shard {
                start_date: 100,
                end_date: 199,
                expected: 20,
            },
        ]
    }

    #[test]
    fn a_fresh_database_has_nothing_to_resume() {
        let (state, _d) = state();
        assert!(state.resumable(1).unwrap().is_none());
    }

    #[test]
    fn an_unfinished_crawl_is_resumable_and_reports_progress() {
        let (state, _d) = state();
        let id = state.begin_crawl(7, 1000, "url", 30).unwrap();
        state.record_shards(id, &shards()).unwrap();

        let pending = state.pending_shards(id).unwrap();
        assert_eq!(pending.len(), 2);
        // Largest windows first, so the long pole starts earliest.
        assert_eq!(pending[0].shard.expected, 20);

        state
            .mark_done(id, pending[0].idx, 20, 1, "Exhausted", Some(555))
            .unwrap();
        let resumable = state.resumable(7).unwrap().unwrap();
        assert_eq!((resumable.done_shards, resumable.total_shards), (1, 2));
        assert_eq!(state.pending_shards(id).unwrap().len(), 1);
        assert!(!state.all_shards_done(id).unwrap());
    }

    #[test]
    fn a_shard_left_running_is_retried_rather_than_trusted() {
        let (state, _d) = state();
        let id = state.begin_crawl(7, 1000, "url", 30).unwrap();
        state.record_shards(id, &shards()).unwrap();
        state.mark_running(id, 0).unwrap();

        // A new process finds it pending again, because its Parquet file has no footer.
        assert_eq!(state.pending_shards(id).unwrap().len(), 2);
    }

    #[test]
    fn a_crawl_is_finished_once_every_shard_has_landed() {
        let (state, _d) = state();
        let id = state.begin_crawl(7, 1000, "url", 30).unwrap();
        state.record_shards(id, &shards()).unwrap();
        state
            .mark_done(id, 0, 10, 1, "Exhausted", Some(400))
            .unwrap();
        assert!(!state.all_shards_done(id).unwrap());

        state
            .mark_done(id, 1, 20, 1, "Exhausted", Some(900))
            .unwrap();
        assert!(state.all_shards_done(id).unwrap());
        state.finish_crawl(id).unwrap();

        assert!(state.resumable(7).unwrap().is_none());
        assert_eq!(state.completed_totals(id).unwrap(), (30, 2));
    }

    /// Minimal scratch directory, removed on drop, so tests never touch a shared path.
    mod tempdir {
        use std::path::{Path, PathBuf};

        #[derive(Debug)]
        pub struct Dir(PathBuf);

        impl Dir {
            pub fn new() -> Self {
                let unique = format!(
                    "census-test-{}-{:?}",
                    std::process::id(),
                    std::thread::current().id()
                );
                let path = std::env::temp_dir().join(unique);
                let _ = std::fs::remove_dir_all(&path);
                std::fs::create_dir_all(&path).unwrap();
                Self(path)
            }

            pub fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }
}
