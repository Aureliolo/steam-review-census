//! Request construction for Valve's `appreviews` endpoint.
//!
//! The parameter choices encoded here are the difference between a census and a sample.
//! Measured against Cyberpunk 2077 (app 1091500), Valve's own defaults return 881,295
//! reviews where an unfiltered request returns 986,295. The 105,000 difference is not
//! evenly drawn: it hides 17.3% of all negative reviews against 9.6% of positive ones,
//! because the two default filters remove activated keys and review bombs respectively.
//! Accepting the defaults biases a corpus against exactly the opinions it exists to count.

use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};

const BASE_URL: &str = "https://store.steampowered.com/appreviews";

/// Valve caps a page at 100 regardless of what is requested; asking for more is silently
/// clamped server-side, so the crawler plans its request budget against this number.
pub const MAX_PER_PAGE: u16 = 100;

/// Cursors are opaque base64 containing `+`, `/` and `=`, all of which change meaning in a
/// query string unless escaped.
const CURSOR_ESCAPE: &AsciiSet = &CONTROLS
    .add(b'+')
    .add(b'/')
    .add(b'=')
    .add(b' ')
    .add(b'&')
    .add(b'#')
    .add(b'?');

/// Pagination order for a review query.
///
/// Valve's `all` ordering is deliberately absent. It ranks by helpfulness and stalls after
/// a handful of pages: a measured crawl of a 2,909-review title returned 21 unique reviews
/// before its cursor began repeating, all drawn from the top of the pile. Only the
/// date-ordered modes walk a corpus to exhaustion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    /// By creation date, newest first. The order used for a full census and for topping up
    /// against a watermark.
    #[default]
    Recent,
    /// By last-edit date, newest first. Used to catch reviews edited since the last crawl.
    Updated,
}

impl SortOrder {
    #[must_use]
    pub fn as_param(self) -> &'static str {
        match self {
            Self::Recent => "recent",
            Self::Updated => "updated",
        }
    }
}

/// A single page request against the `appreviews` endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewQuery {
    app_id: u32,
    order: SortOrder,
    cursor: String,
    per_page: u16,
    window: Option<(i64, i64)>,
}

impl ReviewQuery {
    /// Starts a query at the beginning of a corpus.
    #[must_use]
    pub fn new(app_id: u32) -> Self {
        Self {
            app_id,
            order: SortOrder::default(),
            cursor: "*".to_owned(),
            per_page: MAX_PER_PAGE,
            window: None,
        }
    }

    #[must_use]
    pub fn order(mut self, order: SortOrder) -> Self {
        self.order = order;
        self
    }

    /// Advances to the next page. The cursor is taken verbatim from the previous response.
    #[must_use]
    pub fn cursor(mut self, cursor: impl Into<String>) -> Self {
        self.cursor = cursor.into();
        self
    }

    #[must_use]
    pub fn per_page(mut self, per_page: u16) -> Self {
        self.per_page = per_page.min(MAX_PER_PAGE);
        self
    }

    /// Restricts the query to reviews created within a Unix-timestamp window.
    ///
    /// Date windows are what make a large corpus tractable: they bound how deep any single
    /// cursor walk goes, and they let independent shards be crawled and resumed in
    /// parallel rather than as one 10,000-page sequence.
    #[must_use]
    pub fn window(mut self, start: i64, end: i64) -> Self {
        self.window = Some((start, end));
        self
    }

    #[must_use]
    pub fn to_url(&self) -> String {
        let window = match self.window {
            Some((start, end)) => {
                format!("&start_date={start}&end_date={end}&date_range_type=include")
            }
            None => String::new(),
        };
        format!(
            "{BASE_URL}/{app}?json=1&language=all&purchase_type=all&filter_offtopic_activity=0&filter={filter}&num_per_page={per_page}&cursor={cursor}{window}",
            app = self.app_id,
            filter = self.order.as_param(),
            per_page = self.per_page,
            cursor = utf8_percent_encode(&self.cursor, CURSOR_ESCAPE),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn census_defaults_override_valve_biased_defaults() {
        let url = ReviewQuery::new(1_091_500).to_url();
        assert!(url.contains("purchase_type=all"), "{url}");
        assert!(url.contains("filter_offtopic_activity=0"), "{url}");
        assert!(url.contains("language=all"), "{url}");
    }

    #[test]
    fn pagination_never_uses_helpfulness_ranking() {
        for order in [SortOrder::Recent, SortOrder::Updated] {
            let url = ReviewQuery::new(1).order(order).to_url();
            assert!(!url.contains("filter=all"), "{url}");
        }
        assert!(ReviewQuery::new(1).to_url().contains("filter=recent"));
    }

    #[test]
    fn first_page_uses_the_opening_cursor() {
        assert!(ReviewQuery::new(1).to_url().contains("cursor=*"));
    }

    #[test]
    fn cursor_base64_is_escaped() {
        let url = ReviewQuery::new(1).cursor("AoJ4nczavqADdb+y/wY=").to_url();
        assert!(url.contains("cursor=AoJ4nczavqADdb%2By%2FwY%3D"), "{url}");
    }

    #[test]
    fn per_page_is_clamped_to_what_valve_will_serve() {
        assert!(
            ReviewQuery::new(1)
                .per_page(500)
                .to_url()
                .contains("num_per_page=100")
        );
    }

    #[test]
    fn window_is_omitted_unless_requested() {
        assert!(!ReviewQuery::new(1).to_url().contains("date_range_type"));
        let url = ReviewQuery::new(1)
            .window(1_704_067_200, 1_719_792_000)
            .to_url();
        assert!(url.contains("start_date=1704067200"), "{url}");
        assert!(url.contains("end_date=1719792000"), "{url}");
        assert!(url.contains("date_range_type=include"), "{url}");
    }
}
