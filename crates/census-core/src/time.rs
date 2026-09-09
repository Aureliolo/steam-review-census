//! Turning Steam's Unix timestamps into calendar labels.
//!
//! A dependency-free civil calendar rather than a date crate: what is needed is a UTC day
//! and the month it falls in, which is arithmetic, and a corpus that is only ever labelled
//! by month does not need time zones, parsing, or formatting.

/// The UTC day a timestamp falls in, as year, month and day.
#[must_use]
pub fn civil(unix: i64) -> (i64, u8, u8) {
    civil_from_days(unix.div_euclid(86_400))
}

/// The month a timestamp falls in, as `2024-02`, which sorts as it reads.
#[must_use]
pub fn year_month(unix: i64) -> String {
    let (year, month, _) = civil(unix);
    format!("{year:04}-{month:02}")
}

/// A month label as a short human name, `Feb 2024`.
#[must_use]
pub fn month_name(label: &str) -> String {
    let Some((year, month)) = label.split_once('-') else {
        return label.to_owned();
    };
    let Ok(month) = month.parse::<usize>() else {
        return label.to_owned();
    };
    SHORT
        .get(month.saturating_sub(1))
        .map_or_else(|| label.to_owned(), |name| format!("{name} {year}"))
}

/// A timestamp as a readable day, `14 November 2023`.
#[must_use]
pub fn day(unix: i64) -> String {
    if unix <= 0 {
        return "unknown".to_owned();
    }
    let (year, month, day) = civil(unix);
    let name = LONG
        .get(usize::from(month).saturating_sub(1))
        .copied()
        .unwrap_or("?");
    format!("{day} {name} {year}")
}

const SHORT: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

const LONG: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// Howard Hinnant's days-from-civil, inverted. Exact for every date this tool can hold.
///
/// The single-letter names are the published algorithm's own. Renaming them to something
/// descriptive would only obscure which algorithm this is and make it harder to check.
fn civil_from_days(days: i64) -> (i64, u8, u8) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (
        if m <= 2 { y + 1 } else { y },
        u8::try_from(m).unwrap_or(1),
        u8::try_from(d).unwrap_or(1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_match_the_calendar() {
        assert_eq!(day(0), "unknown");
        assert_eq!(day(1_700_000_000), "14 November 2023");
        // A leap day, which off-by-one arithmetic gets wrong.
        assert_eq!(day(1_709_164_800), "29 February 2024");
        // The last second of a month must not already be the next one.
        assert_eq!(day(1_709_251_199), "29 February 2024");
        assert_eq!(day(1_709_251_200), "1 March 2024");
    }

    #[test]
    fn months_sort_the_way_they_read() {
        let mut months = vec![
            year_month(1_709_164_800),
            year_month(1_667_260_800),
            year_month(1_700_000_000),
        ];
        months.sort();
        assert_eq!(months, vec!["2022-11", "2023-11", "2024-02"]);
    }

    #[test]
    fn month_labels_become_names_and_survive_nonsense() {
        assert_eq!(month_name("2024-02"), "Feb 2024");
        assert_eq!(month_name("2024-12"), "Dec 2024");
        assert_eq!(month_name("nonsense"), "nonsense");
        assert_eq!(month_name("2024-99"), "2024-99");
    }
}
