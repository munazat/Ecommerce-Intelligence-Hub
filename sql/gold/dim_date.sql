-- dim_date: one row per calendar day spanned by the data.
--
-- Includes days with zero transactions on purpose. A continuous date axis is what makes
-- time-series work honest later: a missing day reads as missing demand, not as a missing
-- row, and weekly aggregation for forecasting does not have to invent a calendar.
--
-- Bounds come from the data rather than being hardcoded, so a re-download that extends
-- the history widens the dimension automatically.
--
-- Both day-of-week conventions are exposed because they disagree and mixing them is a
-- classic off-by-one: dayofweek() is 0=Sunday..6=Saturday, isodow() is 1=Monday..7=Sunday.
-- is_weekend uses dayofweek() IN (0, 6). iso_year/iso_week are the correct pairing for
-- weekly revenue series -- e.g. 2011-01-02 belongs to ISO year 2010, week 52, and using
-- the calendar year there would split a single trading week across two years.

CREATE OR REPLACE TABLE dim_date AS
WITH days AS (
    SELECT CAST(date_day AS DATE) AS date_day
    FROM generate_series(
        (SELECT MIN(invoice_date)::DATE FROM fact_order_lines),
        (SELECT MAX(invoice_date)::DATE FROM fact_order_lines),
        INTERVAL 1 DAY
    ) AS t(date_day)
)
SELECT
    CAST(strftime(date_day, '%Y%m%d') AS INTEGER) AS date_key,
    date_day,
    year(date_day)                    AS year,
    quarter(date_day)                 AS quarter,
    month(date_day)                   AS month,
    monthname(date_day)               AS month_name,
    day(date_day)                     AS day_of_month,
    isoyear(date_day)                 AS iso_year,
    week(date_day)                    AS iso_week,
    isodow(date_day)                  AS iso_day_of_week,
    dayofweek(date_day)               AS day_of_week_sun0,
    dayname(date_day)                 AS day_name,
    dayofweek(date_day) IN (0, 6)     AS is_weekend,
    quarter(date_day) = 4             AS is_q4
FROM days;
